"""Run tracking in the T2 RL workflows and tools (model/t2/runs.py; CPU, short runs).

A recording run must compute exactly what the file mode computes and hold the same per-run files as records;
an interrupted and resumed run must equal the uninterrupted one; the readers must give the same content for a
run unit and for a legacy run directory, so summaries built from either are identical.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import pytest
import torch

from model.common import tracking as tr
from model.t2 import runs as t2runs
from model.nec_workflow import make_nec_config, train_nec, load_nec_checkpoint
from model.nnknn_rl_workflow import make_nnknn_rl_config, train_nnknn_rl, load_nnknn_rl_checkpoint
from model.ppo_workflow import make_ppo_config, train_ppo, load_ppo_checkpoint
from model.rl_workflow import make_dqn_config, train_dqn, load_dqn_checkpoint
from model.td3_workflow import make_td3_config, train_td3, load_td3_checkpoint

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "dqn": (train_dqn, "cartpole", lambda: make_dqn_config("smoke", seed=3, total_timesteps=300, eval_frequency=100,
                                                          learning_starts=40)),
    "nec": (train_nec, "cartpole", lambda: make_nec_config("smoke", seed=4, total_timesteps=300)),
    "ppo": (train_ppo, "cartpole", lambda: make_ppo_config("smoke", seed=5, total_timesteps=400, eval_frequency=150)),
    "ppo_continuous": (train_ppo, "pendulum", lambda: make_ppo_config("smoke", seed=6, total_timesteps=420,
                                                                      eval_frequency=200)),
    "td3": (train_td3, "pendulum", lambda: make_td3_config("smoke", seed=7, total_timesteps=420)),
    "nnknn": (train_nnknn_rl, "cartpole", lambda: make_nnknn_rl_config("smoke", seed=8, total_timesteps=300)),
}


def nnknn_full_config(steps=384):
    """Both roles NN-kNN with every per-run file switched on (event JSONs, retention and audit snapshots)."""
    return make_nnknn_rl_config(
        "smoke", seed=9, total_timesteps=steps, actor_type="nnknn", critic_type="nnknn", case_capacity=64,
        critic_case_capacity=64, policy_update_episodes=2, eval_frequency=0, eval_episode_frequency=4,
        eval_episodes=2, critic_holdout_episode_frequency=4, critic_holdout_episodes=2, early_stopping=False,
        success_threshold=None, top_k=8, min_case_entries=8, min_cases_per_action=2,
        case_optimizer_maintenance="preserve_by_id", case_quality_tracking=True, case_maintenance_frequency=100,
        case_retention_keep_fraction=0.9, case_retention_frequency=128, case_retention_queries=8,
        case_audit_queries_per_batch=2, critic_mutable_value_labels=True, critic_trainable_value_labels=True)


# paths, file hashes of snapshots, wall-clock fields and the ledger's event digests (which hash snapshot paths)
VOLATILE = {"snapshot", "snapshot_sha256", "last_snapshot_sha256", "selection_seconds", "namespace",
            "created_at_utc", "run_dir", "checkpoint_path"}


def normalize(obj):
    if isinstance(obj, dict):
        return {k: (sorted(v) if k == "events" and isinstance(v, dict) else normalize(v))
                for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    return obj


def digest(obj):
    return tr.state_digest(t2runs.tensorize(normalize(obj)))


@pytest.fixture
def root(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    monkeypatch.setenv("NNKNN_RESUME_EVERY_SECONDS", "0")
    monkeypatch.setenv("NNKNN_RESUME_EVERY_STEPS", "60")
    monkeypatch.delenv("NNKNN_RESUME", raising=False)
    return runs


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Crash(Exception):
    pass


def record(name, train, task, cfg, *, crash_after=None, study="t", config=None):
    """Train in unit t2/<study>/<name>; with crash_after=n the first session dies right after its n-th checkpoint
    and a second session resumes. Returns (unit path, returned state of the last session, resumed?)."""
    key = tr.RunKey("t2", study, (name,))
    config = config or {"case": "same"}
    if crash_after is not None:
        original, seen = tr.Run.save_resume, []

        def save_resume(self, state, *, step, generators=None):
            out = original(self, state, step=step, generators=generators)
            seen.append(step)
            if len(seen) == crash_after:
                raise Crash(step)
            return out

        tr.Run.save_resume = save_resume
        try:
            with pytest.raises(Crash):
                with tr.open_run(key, config=config) as run:
                    train(task, cfg(), device="cpu", progress=False, run=run, resumable=True)
        finally:
            tr.Run.save_resume = original
    with tr.open_run(key, config=config) as run:
        resumed = run.resumed_from is not None
        state = train(task, cfg(), device="cpu", progress=False, run=run, resumable=True)
    return tr.unit_dir(key), state, resumed


def assert_same_files(run_dir: Path, unit: Path):
    """Every file of the file-mode run, rebuilt from the unit's records, has the same content."""
    files = t2runs.RunFiles(unit)
    legacy = t2runs.RunFiles(run_dir)
    names = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
    assert names, run_dir
    for rel in names:
        if rel == "manifest.json":
            assert files.json(rel)["outputs"]
        elif rel.endswith(".pt"):
            assert digest(tr.load_state(unit, t2runs.state_name(rel))) == digest(
                torch.load(run_dir / rel, map_location="cpu", weights_only=False)), rel
        elif rel.endswith(".json"):
            assert normalize(json.loads(json.dumps(files.json(rel)))) == normalize(legacy.json(rel)), rel
        else:
            assert files.text(rel) == legacy.text(rel), rel  # CSVs byte for byte
            assert files.frame(rel).equals(legacy.frame(rel)) if legacy.text(rel) else True
    # the recording run wrote nothing but event files and weights
    assert {p.parent.name if p.parent != unit else "" for p in unit.rglob("*") if p.is_file()
            and "tfevents" not in p.name} <= {"weights", "resume"}


def logical_stream(unit):
    return [(r.tag, r.step, r.raw) for r in tr._scan(unit).stream if not r.tag.startswith("rec/state/")]


@pytest.mark.parametrize("name", sorted(CASES))
def test_recording_computes_the_same_and_holds_the_same_files(root, tmp_path, name):
    train, task, cfg = CASES[name]
    plain = train(task, cfg(), output_dir=tmp_path / "plain", device="cpu", progress=False)
    unit, recorded, _ = record(name, train, task, cfg)
    assert tr.run_status(unit).status == "complete"
    for rows in ("training_metrics", "loss_metrics", "eval_metrics", "final_eval"):
        assert plain[rows] == recorded[rows], rows
    assert_same_files(tmp_path / "plain", unit)
    data = tr.load_run(unit)
    assert data.final()["selected_mean_return"] == plain["final_eval"]["mean_return"]
    assert data.series("episode/return")[0][1] == plain["training_metrics"][0]["episode_return"]
    assert recorded["summary"]["run_dir"] == "." and recorded["summary"]["checkpoint_path"] == "weights/checkpoint.pt"
    assert t2runs.RunFiles(unit).json("summary.json")["run_dir"] == str(unit)


@pytest.mark.parametrize("name", sorted(CASES))
def test_interrupted_and_resumed_equals_uninterrupted(root, name):
    train, task, cfg = CASES[name]
    whole, state_a, _ = record(name, train, task, cfg, study="whole")
    resumed, state_b, was_resumed = record(name, train, task, cfg, study="cut", crash_after=2)
    assert was_resumed and tr.run_status(resumed).sessions == 2
    assert tr.run_status(whole).result_digest == tr.run_status(resumed).result_digest
    assert ({k: v["state_sha256"] for k, v in tr.load_run(whole).states().items()}
            == {k: v["state_sha256"] for k, v in tr.load_run(resumed).states().items()})
    for rows in ("training_metrics", "loss_metrics", "eval_metrics"):
        assert state_a[rows] == state_b[rows]


def test_nnknn_with_every_per_run_file(root, tmp_path):
    """Event JSONs, retention and case-audit snapshots: same content; resumed equals uninterrupted except the
    wall-clock selection_seconds of the retention events."""
    plain = train_nnknn_rl("cartpole", nnknn_full_config(), output_dir=tmp_path / "plain", device="cpu",
                           progress=False)
    whole, state_a, _ = record("full", train_nnknn_rl, "cartpole", nnknn_full_config, study="whole")
    assert_same_files(tmp_path / "plain", whole)
    assert plain["loss_metrics"] == state_a["loss_metrics"]
    states = tr.load_run(whole).states()
    assert any(k.startswith("retention_states.") for k in states)
    assert any(k.startswith("case_audit_states.") for k in states)
    files = t2runs.RunFiles(whole)
    for event in files.json("case_audit_events.json")["events"]:  # snapshots recorded relative to the unit
        assert event["snapshot"].startswith("weights/case_audit_states.")
        path = Path(files.resolve(event["snapshot"]))
        assert path.is_file() and tr._file_sha256(path)[0] == event["snapshot_sha256"]

    resumed, state_b, was_resumed = record("full", train_nnknn_rl, "cartpole", nnknn_full_config, study="cut",
                                           crash_after=2)
    assert was_resumed
    a, b = logical_stream(whole), logical_stream(resumed)
    assert [x[:2] for x in a] == [x[:2] for x in b]
    for x, y in zip(a, b):
        if x != y:
            assert x[0] == "rec/files/case_retention_events.json/events"
            dx, dy = (json.loads(v) for v in (x[2], y[2]))
            for row in (*dx["rows"], *dy["rows"]):
                row.pop("selection_seconds")
            assert dx == dy
    assert ({k: v["state_sha256"] for k, v in tr.load_run(whole).states().items()}
            == {k: v["state_sha256"] for k, v in tr.load_run(resumed).states().items()})


KILL = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(1)  # as the test process (the fixture one_thread)
from model.common import tracking as tr
from model.rl_workflow import make_dqn_config, train_dqn
original = tr.Run.save_resume
def save_resume(self, state, *, step, generators=None):
    out = original(self, state, step=step, generators=generators)
    if step >= int(sys.argv[2]):
        os._exit(9)  # no _run/failed, no flush of anything after the checkpoint
    return out
tr.Run.save_resume = save_resume
with tr.open_run(tr.RunKey("t2", "kill", ("dqn",)), config={"case": "same"}) as run:
    train_dqn("cartpole", make_dqn_config("smoke", seed=3, total_timesteps=300, eval_frequency=100, learning_starts=40),
              device="cpu", progress=False, run=run, resumable=True)
"""


def test_a_killed_process_resumes_exactly(root, tmp_path):
    train, task, cfg = CASES["dqn"]
    whole, _, _ = record("dqn", train, task, cfg, study="whole")
    script = tmp_path / "kill.py"
    script.write_text(KILL, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(ROOT), "120"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert proc.returncode == 9, proc.stderr[-2000:]
    unit = tr.unit_dir(tr.RunKey("t2", "kill", ("dqn",)))
    assert tr.run_status(unit).status == "incomplete"
    with tr.open_run(tr.RunKey("t2", "kill", ("dqn",)), config={"case": "same"}) as run:
        assert run.resumed_from is not None
        train(task, cfg(), device="cpu", progress=False, run=run, resumable=True)
    assert tr.run_status(unit).result_digest == tr.run_status(whole).result_digest


def test_resumable_needs_a_recording_run_and_a_resume_needs_resumable(root, tmp_path):
    train, task, cfg = CASES["dqn"]
    with pytest.raises(ValueError, match="recording run"):
        train(task, cfg(), output_dir=tmp_path / "x", device="cpu", progress=False, resumable=True)
    key = tr.RunKey("t2", "t", ("half",))
    original = tr.Run.save_resume

    def save_resume(self, state, *, step, generators=None):
        original(self, state, step=step, generators=generators)
        raise Crash(step)

    tr.Run.save_resume = save_resume
    try:
        with pytest.raises(Crash), tr.open_run(key, config={}) as run:
            train(task, cfg(), device="cpu", progress=False, run=run, resumable=True)
    finally:
        tr.Run.save_resume = original
    with pytest.raises(RuntimeError, match="resumable=False"), tr.open_run(key, config={}) as run:
        train(task, cfg(), device="cpu", progress=False, run=run)


def test_null_run_and_none_write_the_files_as_before(root, tmp_path):
    train, task, cfg = CASES["dqn"]
    a = train(task, cfg(), output_dir=tmp_path / "a", device="cpu", progress=False)
    b = train(task, cfg(), output_dir=tmp_path / "b", device="cpu", progress=False, run=tr.NullRun())
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "b").iterdir()) == sorted(
        ["config.json", "training_metrics.csv", "loss_metrics.csv", "eval_metrics.csv", "final_eval_episodes.csv",
         "last_eval_episodes.csv", "summary.json", "manifest.json", "checkpoint.pt"])
    for name in names:
        if name.endswith(".csv"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert a["summary"]["run_dir"] == str(tmp_path / "a") and b["summary"]["run_dir"] == str(tmp_path / "b")
    assert a["loss_metrics"] == b["loss_metrics"]
    assert not any(root.iterdir())


@pytest.mark.parametrize("name,loader", [("dqn", load_dqn_checkpoint), ("nec", load_nec_checkpoint),
                                         ("ppo", load_ppo_checkpoint), ("td3", load_td3_checkpoint),
                                         ("nnknn", load_nnknn_rl_checkpoint)])
def test_checkpoint_loaders_read_a_run_unit(root, tmp_path, name, loader):
    train, task, cfg = CASES[name]
    train(task, cfg(), output_dir=tmp_path / "plain", device="cpu", progress=False)
    unit, _, _ = record(name, train, task, cfg)
    a, b = loader(tmp_path / "plain" / "checkpoint.pt", device="cpu"), loader(unit, device="cpu")
    assert tr.state_digest(a["model"].state_dict()) == tr.state_digest(b["model"].state_dict())
    assert a["config"] == b["config"]
    if name == "nec":
        assert digest(a["dnd"].to_state()) == digest(b["dnd"].to_state())


def test_phase4_report_is_byte_identical_from_run_units(root, tmp_path, monkeypatch):
    """The same runs as run directories and as run units give the same table and the same PNG files."""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_phase4_report as rep

    manifests = {"dirs": {}, "units": {}}
    for method, (train, cfgf) in {"dqn": (train_dqn, CASES["dqn"][2]), "ppo": (train_ppo, CASES["ppo"][2])}.items():
        for seed in (0, 1):
            def cfg(cfgf=cfgf, seed=seed):
                return dataclasses.replace(cfgf(), seed=seed)
            d = tmp_path / "dirs" / f"{method}{seed}"
            train("cartpole", cfg(), output_dir=d, device="cpu", progress=False)
            unit, _, _ = record(f"{method}{seed}", train, "cartpole", cfg, study="phase4")
            manifests["dirs"].setdefault(method, {})[str(seed)] = str(d)
            manifests["units"].setdefault(method, {})[str(seed)] = str(unit)
    outputs = {}
    for kind, manifest in manifests.items():
        data = rep.load_runs(manifest)
        order = rep.resolve_method_order(manifest)
        kwargs = dict(task="CartPole-v1", threshold=50.0, ylim=(0.0, 525.0), prefix="p", order=order)
        outdir = tmp_path / f"fig_{kind}"
        outdir.mkdir()
        table = rep.build_table(data, 50.0, order)
        pngs = [rep.plot_per_method(data, outdir, **kwargs), rep.plot_means(data, outdir, **kwargs)]
        outputs[kind] = (table, [Path(p).read_bytes() for p in pngs])
    assert outputs["dirs"] == outputs["units"]


def test_run_rl_tool_study_records_a_unit_and_skips_it_when_complete(root, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_dqn

    argv = ["run_rl_dqn.py", "cartpole", "--profile", "smoke", "--seed", "2", "--total-timesteps", "200",
            "--device", "cpu", "--quiet", "--output-dir", str(tmp_path / "results")]
    monkeypatch.setattr(sys, "argv", argv)
    run_rl_dqn.main()
    plain = capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", argv + ["--study", "tool"])
    run_rl_dqn.main()
    first = capsys.readouterr().out
    run_rl_dqn.main()
    second = capsys.readouterr().out
    unit = root / "t2" / "tool" / "dqn" / "cartpole" / "seed2"
    assert tr.run_status(unit).status == "complete" and tr.run_status(unit).sessions == 1
    assert "complete, not rerun" in second and "fresh" in first

    def printed(text):
        return json.loads(text[text.index("{"):text.rindex("}") + 1])

    assert printed(first) == printed(second)
    assert normalize(printed(plain)) == normalize(printed(first))
    assert len(list((tmp_path / "results").iterdir())) == 1  # --study made no run directory
    # eval-only reads the unit as a checkpoint
    monkeypatch.setattr(sys, "argv", ["run_rl_dqn.py", "--eval-only", "--checkpoint", str(unit), "--device", "cpu",
                                      "--output-dir", str(tmp_path / "eval")])
    run_rl_dqn.main()
    assert "mean_return" in capsys.readouterr().out


def test_run_rl_tool_checks_the_study_before_training(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_dqn

    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "missing"))
    monkeypatch.setattr(sys, "argv", ["run_rl_dqn.py", "--profile", "smoke", "--study", "x"])
    with pytest.raises(SystemExit, match="--study"):
        run_rl_dqn.main()


def test_battery_rows_from_units_equal_rows_from_run_directories(root, tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_mcb_and_capacity_experiments as mcb

    kw = dict(task="cartpole", seed=1, actor_type="nnknn", critic_type="mlp", case_capacity=64, profile="smoke",
              device="cpu")
    a = mcb.run_single_experiment(output_dir=tmp_path / "battery", **kw)
    b = mcb.run_single_experiment(output_dir=tmp_path / "battery", study="mcb", **kw)
    c = mcb.run_single_experiment(output_dir=tmp_path / "battery", study="mcb", **kw)  # complete: read back
    unit = root / "t2" / "mcb" / "battery" / "cartpole_nnknn_mlp_cap64_s1"
    assert b["run_dir"] == c["run_dir"] == str(unit)
    assert b["elapsed_seconds"] == c["elapsed_seconds"] == t2runs.RunFiles(unit).train_seconds()
    for row in (a, b, c):
        row.pop("run_dir"), row.pop("elapsed_seconds")
    assert a == b == c


def test_cartpole_sweep_study_rows(root, tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_cartpole_all_variants as sweep

    base = ["x", "--profile", "smoke", "--variants", "dqn,nnknn_actor_mlp_critic", "--device", "cpu",
            "--total-timesteps", "200", "--eval-episodes", "2"]
    rows = {}
    for kind, extra in (("dirs", ["--output-dir", str(tmp_path / "a")]),
                        ("units", ["--output-dir", str(tmp_path / "b"), "--study", "sweep"]),
                        ("again", ["--output-dir", str(tmp_path / "c"), "--study", "sweep"])):
        monkeypatch.setattr(sys, "argv", base + extra)
        sweep.main()
        out = next(Path(extra[1]).iterdir())
        rows[kind] = json.loads((out / "sweep_summary.json").read_text(encoding="utf-8"))["rows"]
    assert [r["status"] for r in rows["units"]] == ["ok", "ok"]
    assert [r["run_dir"] for r in rows["units"]] == [r["run_dir"] for r in rows["again"]]
    for kind in rows:
        for r in rows[kind]:
            r.pop("run_dir"), r.pop("elapsed_seconds")
    assert rows["dirs"] == rows["units"] == rows["again"]


def test_role_pilot_study_matches_the_run_folder_records(root, tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import t2_role_pilot as pilot

    base = ["x", "--steps", "256", "--seeds", "8", "--roles", "nnknn_nnknn", "--audit-queries-per-batch", "2"]
    manifests = {}
    for kind, extra in (("dirs", ["--output", str(tmp_path / "a")]),
                        ("units", ["--output", str(tmp_path / "b"), "--study", "pilot"]),
                        ("again", ["--output", str(tmp_path / "c"), "--study", "pilot"])):
        monkeypatch.setattr(sys, "argv", base + extra)
        pilot.main()
        manifests[kind] = json.loads((Path(extra[1]) / "manifest.json").read_text(encoding="utf-8"))["roles"]
    unit = root / "t2" / "pilot" / "cartpole" / "nnknn_nnknn__ema__reset" / "s8"
    assert manifests["units"][0]["run_dir"] == str(unit) and tr.run_status(unit).sessions == 1
    assert manifests["units"][0]["checkpoint_sha256"] == tr._file_sha256(unit / "weights" / "checkpoint.pt")[0]
    assert manifests["units"] == manifests["again"]
    for kind in manifests:
        for r in manifests[kind]:
            for k in ("run_dir", "training_seconds", "checkpoint_sha256"):
                r.pop(k)
    assert normalize(manifests["dirs"]) == normalize(manifests["units"])
