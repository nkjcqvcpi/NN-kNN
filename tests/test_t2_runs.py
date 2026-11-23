"""Run tracking in the T2 RL workflows and tools (model/t2/runs.py; CPU, short runs).

Recording is mandatory (owner's rules O1-O5, 2026-11-22): every training needs an open run and writes nothing but
the unit; tools need ``--study``; readers read only run units; a complete unit is never trained (or evaluated, or
computed) again, and a partly rerun study equals an uninterrupted one. An interrupted and resumed run must equal
the uninterrupted one.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import types
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


BASE_FILES = ["config.json", "training_metrics.csv", "loss_metrics.csv", "eval_metrics.csv",
              "final_eval_episodes.csv", "last_eval_episodes.csv", "summary.json", "checkpoint.pt"]


def assert_unit_only(unit: Path):
    """The run wrote nothing but event files and weights (no per-run file of the retired file mode)."""
    assert {p.parent.name if p.parent != unit else "" for p in unit.rglob("*") if p.is_file()
            and "tfevents" not in p.name} <= {"weights", "resume"}


def logical_stream(unit):
    """The unit's logical stream without state bindings and the wall-clock ``x/sys/*`` values."""
    return [(r.tag, r.step, r.raw) for r in tr._scan(unit).stream
            if not r.tag.startswith(("rec/state/", "x/sys/"))]


QUALITY_IDENTITY = ("namespace", "events", "last_snapshot_sha256")


def quality_numbers(state):
    """A quality-ledger state without its path/serialisation identities (namespace, event digests, snapshot
    hashes): the rows' C/H/Q/exposure and provenance steps, and the event ids."""
    return {"event_ids": sorted(state["events"]), "semantics": state["semantics"],
            "rows": [{k: v for k, v in row.items() if k not in QUALITY_IDENTITY} for row in state["rows"]]}


@pytest.mark.parametrize("name", sorted(CASES))
def test_recording_holds_every_per_run_file(root, name):
    train, task, cfg = CASES[name]
    unit, recorded, _ = record(name, train, task, cfg)
    assert tr.run_status(unit).status == "complete"
    files = t2runs.RunFiles(unit)
    assert set(BASE_FILES) <= set(files.names())
    for rows in ("training_metrics", "loss_metrics", "eval_metrics"):
        assert files.rows(f"{rows}.csv") == recorded[rows], rows
        if recorded[rows]:
            assert files.frame(f"{rows}.csv").shape[0] == len(recorded[rows])
    assert normalize(files.json("summary.json")) == normalize(json.loads(json.dumps(recorded["summary"])))
    assert_unit_only(unit)
    data = tr.load_run(unit)
    assert data.final()["selected_mean_return"] == recorded["final_eval"]["mean_return"]
    assert data.series("episode/return")[0][1] == recorded["training_metrics"][0]["episode_return"]
    assert recorded["summary"]["run_dir"] == "." and recorded["summary"]["checkpoint_path"] == "weights/checkpoint.pt"
    assert files.json("summary.json")["run_dir"] == str(unit)
    assert recorded["run_dir"] == unit and recorded["checkpoint_path"] == files.checkpoint_path()


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


def test_nnknn_with_every_per_run_file(root):
    """Event JSONs, retention and case-audit snapshots are records and named states; resumed equals uninterrupted
    except the wall-clock selection_seconds of the retention events."""
    whole, state_a, _ = record("full", train_nnknn_rl, "cartpole", nnknn_full_config, study="whole")
    assert_unit_only(whole)
    states = tr.load_run(whole).states()
    assert any(k.startswith("retention_states.") for k in states)
    assert any(k.startswith("case_audit_states.") for k in states)
    files = t2runs.RunFiles(whole)
    assert {"case_optimizer_events.json", "case_audit_events.json", "case_retention_events.json",
            "case_quality_selected.json", "critic_holdout_metrics.csv", "case_maintenance.csv"} <= set(files.names())
    assert files.rows("loss_metrics.csv") == state_a["loss_metrics"]
    # the quality ledger's namespace is the unit itself; the selected ledger equals the checkpoint's
    unit_ckpt = tr.load_state(whole, "checkpoint")
    ledger = files.json("case_quality_selected.json")
    assert ledger["rows"] and ledger["namespace"] == "."
    assert quality_numbers(unit_ckpt["selected_case_quality_state"]) == quality_numbers(ledger)
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


@pytest.mark.parametrize("name", sorted(CASES))
def test_training_requires_an_open_run(root, name):
    train, task, cfg = CASES[name]
    for run in (None, tr.NullRun()):
        with pytest.raises(TypeError, match="open tracking.Run"):
            train(task, cfg(), device="cpu", progress=False, run=run)
    with pytest.raises(TypeError):
        train(task, cfg(), device="cpu", progress=False)  # run is a required argument
    assert not any(root.iterdir())


def test_a_resume_needs_resumable(root):
    train, task, cfg = CASES["dqn"]
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


def test_readers_read_only_run_units(tmp_path):
    legacy = tmp_path / "dqn_cartpole_legacy"
    legacy.mkdir()
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not a run unit"):
        t2runs.RunFiles(legacy)


@pytest.mark.parametrize("name,loader", [("dqn", load_dqn_checkpoint), ("nec", load_nec_checkpoint),
                                         ("ppo", load_ppo_checkpoint), ("td3", load_td3_checkpoint),
                                         ("nnknn", load_nnknn_rl_checkpoint)])
def test_checkpoint_loaders_read_a_run_unit(root, name, loader):
    train, task, cfg = CASES[name]
    unit, state, _ = record(name, train, task, cfg)
    a, b = loader(unit, device="cpu"), loader(t2runs.RunFiles(unit).checkpoint_path(), device="cpu")
    assert tr.state_digest(a["model"].state_dict()) == tr.state_digest(b["model"].state_dict())
    assert tr.state_digest(a["model"].state_dict()) == tr.state_digest(state["model"].state_dict())
    assert a["config"] == b["config"]
    if name == "nec":
        assert digest(a["dnd"].to_state()) == digest(b["dnd"].to_state())


def test_phase4_report_reads_only_run_units(root, tmp_path):
    """The report reads summary.json and eval_metrics.csv from run units; a legacy run directory is an error;
    its figures go to <worktree>/results/figures by default."""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_phase4_report as rep

    assert rep.DEFAULT_OUTDIR == ROOT / "results" / "figures"
    manifest = {}
    for method, (train, cfgf) in {"dqn": (train_dqn, CASES["dqn"][2]), "ppo": (train_ppo, CASES["ppo"][2])}.items():
        for seed in (0, 1):
            def cfg(cfgf=cfgf, seed=seed):
                return dataclasses.replace(cfgf(), seed=seed)
            unit, _, _ = record(f"{method}{seed}", train, "cartpole", cfg, study="phase4")
            manifest.setdefault(method, {})[str(seed)] = str(unit)
    data = rep.load_runs(manifest)
    order = rep.resolve_method_order(manifest)
    assert order == ["dqn", "ppo"]
    files = t2runs.RunFiles(manifest["dqn"]["0"])
    assert data["dqn"][0]["summary"] == files.json("summary.json")
    assert data["dqn"][0]["evals"].equals(files.frame("eval_metrics.csv"))
    table = rep.build_table(data, 50.0, order)
    assert table.count("\n") == 3
    outdir = tmp_path / "fig"
    outdir.mkdir()
    kwargs = dict(task="CartPole-v1", threshold=50.0, ylim=(0.0, 525.0), prefix="p", order=order)
    for png in (rep.plot_per_method(data, outdir, **kwargs), rep.plot_means(data, outdir, **kwargs)):
        assert Path(png).read_bytes()[:4] == b"\x89PNG"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a run unit"):
        rep.load_runs({"dqn": {"0": str(legacy)}})


def printed(text):
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def test_run_rl_tool_requires_a_study_records_a_unit_and_never_retrains_it(root, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_dqn

    argv = ["run_rl_dqn.py", "cartpole", "--profile", "smoke", "--seed", "2", "--total-timesteps", "200",
            "--device", "cpu", "--quiet"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        run_rl_dqn.main()
    assert exc.value.code == 2 and "--study" in capsys.readouterr().err
    assert not any(root.iterdir())
    monkeypatch.setattr(sys, "argv", argv + ["--study", "tool"])
    run_rl_dqn.main()
    first = capsys.readouterr().out
    calls = []
    monkeypatch.setattr(run_rl_dqn, "train_dqn", lambda *a, **k: calls.append(1))
    run_rl_dqn.main()  # complete: read back before any training
    second = capsys.readouterr().out
    assert not calls
    unit = root / "t2" / "tool" / "dqn" / "cartpole" / "seed2"
    assert tr.run_status(unit).status == "complete" and tr.run_status(unit).sessions == 1
    assert "complete, not rerun" in second and "fresh" in first
    assert printed(first) == printed(second)
    assert sorted(p.name for p in (root / "t2" / "tool").iterdir()) == ["dqn"]

    # eval-only evaluates the unit's weights and records the evaluation as a unit; a rerun reads it back
    eval_argv = ["run_rl_dqn.py", "--eval-only", "--checkpoint", str(unit), "--device", "cpu", "--study", "tool"]
    monkeypatch.setattr(sys, "argv", eval_argv)
    run_rl_dqn.main()
    metrics = printed(capsys.readouterr().out)
    evaluated = []
    monkeypatch.setattr(run_rl_dqn, "evaluate_dqn", lambda *a, **k: evaluated.append(1))
    run_rl_dqn.main()
    assert printed(capsys.readouterr().out) == metrics and not evaluated
    (eval_unit,) = [i.path for i in tr.iter_runs(root / "t2" / "tool" / "eval")]
    data = tr.load_run(eval_unit)
    assert data.record("files/eval_summary.json")["metrics"] == metrics
    assert data.final()["mean_return"] == metrics["mean_return"]
    assert len(data.records("files/eval_episodes.csv")) == metrics["episodes"]
    monkeypatch.setattr(sys, "argv", ["run_rl_dqn.py", "--eval-only", "--checkpoint",
                                      str(t2runs.RunFiles(unit).checkpoint_path()), "--study", "tool"])
    with pytest.raises(SystemExit, match="not a run unit"):
        run_rl_dqn.main()


def test_run_rl_tool_checks_the_study_before_training(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_dqn

    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "missing"))
    monkeypatch.setattr(sys, "argv", ["run_rl_dqn.py", "--profile", "smoke", "--study", "x"])
    with pytest.raises(SystemExit, match="--study"):
        run_rl_dqn.main()


def test_battery_rows_from_units_and_a_complete_unit_is_read_back(root, tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import run_rl_mcb_and_capacity_experiments as mcb

    assert mcb.DEFAULT_OUTPUT_DIR.is_relative_to(ROOT / "results")
    kw = dict(task="cartpole", seed=1, actor_type="nnknn", critic_type="mlp", case_capacity=64, profile="smoke",
              device="cpu")
    with pytest.raises(TypeError):
        mcb.run_single_experiment(output_dir=tmp_path / "battery", **kw)  # study is required
    b = mcb.run_single_experiment(output_dir=tmp_path / "battery", study="mcb", **kw)
    c = mcb.run_single_experiment(output_dir=tmp_path / "battery", study="mcb", **kw)  # complete: read back
    unit = root / "t2" / "mcb" / "battery" / "cartpole_nnknn_mlp_cap64_s1"
    assert b == c and b["run_dir"] == str(unit) and tr.run_status(unit).sessions == 1
    assert b["elapsed_seconds"] == t2runs.RunFiles(unit).train_seconds()
    assert not (tmp_path / "battery").exists()  # no run directory


def crash_in(unit_name, after=1):
    """Patch Run.save_resume so that the session of the unit whose path ends in ``unit_name`` dies right after its
    ``after``-th checkpoint (once). Returns the restore function."""
    original, seen = tr.Run.save_resume, []

    def save_resume(self, state, *, step, generators=None):
        out = original(self, state, step=step, generators=generators)
        if self.path.name == unit_name and not seen[after - 1:]:
            seen.append(step)
            if len(seen) == after:
                raise Crash(step)
        return out

    tr.Run.save_resume = save_resume
    return lambda: setattr(tr.Run, "save_resume", original)


def states_of(unit):
    return {k: v["state_sha256"] for k, v in tr.load_run(unit).states().items()}


def test_cartpole_sweep_partly_rerun_equals_uninterrupted(root, tmp_path, monkeypatch):
    """O4: the sweep crashes inside its second variant; the rerun skips the complete units before any training
    (the trainer is not called for them), resumes the interrupted one, and its rows and weights equal an
    uninterrupted sweep's."""
    sys.path.insert(0, str(ROOT / "tools"))
    import run_cartpole_all_variants as sweep

    base = ["x", "--profile", "smoke", "--variants", "dqn,nnknn_actor_mlp_critic,mlp_actor_mlp_critic",
            "--device", "cpu", "--total-timesteps", "200", "--eval-episodes", "2"]

    def run(study, out):
        monkeypatch.setattr(sys, "argv", base + ["--output-dir", str(tmp_path / out), "--study", study])
        sweep.main()
        folder = next((tmp_path / out).iterdir())
        return json.loads((folder / "sweep_summary.json").read_text(encoding="utf-8"))["rows"]

    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit):
        sweep.main()  # --study is required
    straight = run("straight", "a")
    order = [r["variant"] for r in straight]
    assert len(order) == 3
    restore = crash_in(order[1])
    try:
        crashed = run("partial", "b")
    finally:
        restore()
    assert [r["status"] for r in crashed] == ["ok", "error", "ok"]
    trained = []
    real = sweep.train_nnknn_rl, sweep.train_dqn
    monkeypatch.setattr(sweep, "train_nnknn_rl", lambda *a, **k: trained.append(k["run"].path.name) or real[0](*a, **k))
    monkeypatch.setattr(sweep, "train_dqn", lambda *a, **k: trained.append(k["run"].path.name) or real[1](*a, **k))
    rerun = run("partial", "c")
    assert trained == [order[1]]  # complete units: no training
    base_dir = root / "t2" / "partial" / "cartpole_all_variants_smoke"
    assert [tr.run_status(base_dir / v).sessions for v in order] == [1, 2, 1]
    for name in order:
        a, b = root / "t2" / "straight" / "cartpole_all_variants_smoke" / name, base_dir / name
        assert tr.run_status(a).result_digest == tr.run_status(b).result_digest, name
        assert states_of(a) == states_of(b), name
    for rows in (straight, rerun):
        for r in rows:
            r.pop("run_dir"), r.pop("elapsed_seconds")
    assert straight == rerun


ROLE_BASE = ["x", "--steps", "256", "--seeds", "8", "9", "--roles", "nnknn_nnknn", "--audit-queries-per-batch", "2"]


def role_manifest(folder):
    return json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))["roles"]


def test_role_pilot_partly_rerun_equals_uninterrupted(root, tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import t2_role_pilot as pilot

    monkeypatch.setattr(sys, "argv", ROLE_BASE + ["--output", str(tmp_path / "a"), "--study", "straight"])
    pilot.main()
    restore = crash_in("s9")
    try:
        monkeypatch.setattr(sys, "argv", ROLE_BASE + ["--output", str(tmp_path / "b"), "--study", "partial"])
        with pytest.raises(Crash):
            pilot.main()
    finally:
        restore()
    trained = []
    real = pilot.train_nnknn_rl
    monkeypatch.setattr(pilot, "train_nnknn_rl", lambda *a, **k: trained.append(k["run"].path.name) or real(*a, **k))
    monkeypatch.setattr(sys, "argv", ROLE_BASE + ["--output", str(tmp_path / "c"), "--study", "partial"])
    pilot.main()
    assert trained == ["s9"]
    straight, rerun = role_manifest(tmp_path / "a"), role_manifest(tmp_path / "c")
    for r in rerun:
        assert tr.run_status(r["run_dir"]).status == "complete"
        assert r["checkpoint_sha256"] == tr._file_sha256(Path(r["run_dir"]) / "weights" / "checkpoint.pt")[0]
    for a, b in zip(straight, rerun):
        assert states_of(a["run_dir"]) == states_of(b["run_dir"])
        # the result digest also covers the role_audit record's wall-clock training_seconds: compare without it
        ra, rb = (tr.load_run(x["run_dir"]) for x in (a, b))
        assert ({k: v for k, v in ra.record("role_audit").items() if k != "training_seconds"}
                == {k: v for k, v in rb.record("role_audit").items() if k != "training_seconds"})
        assert ra.records("role_audit/events") == rb.records("role_audit/events")
        assert ra.records("files/training_metrics.csv") == rb.records("files/training_metrics.csv")
    for rows in (straight, rerun):
        for r in rows:
            for k in ("run_dir", "training_seconds", "checkpoint_sha256"):
                r.pop(k)
    assert normalize(straight) == normalize(rerun)
    monkeypatch.setattr(sys, "argv", ROLE_BASE + ["--output", str(tmp_path / "d")])
    with pytest.raises(SystemExit):
        pilot.main()  # --study is required


def test_retention_pilot_partly_rerun_equals_uninterrupted(root, tmp_path, monkeypatch):
    """O4 with global RNG: a complete condition is not recomputed, and the RNG state it left (rng_after) is
    restored, so the conditions after it give what an uninterrupted pilot gives."""
    sys.path.insert(0, str(ROOT / "tools"))
    import t2_role_pilot as role_pilot
    import t2_retention_pilot as pilot
    import model.t2.retention as retention

    monkeypatch.setattr(sys, "argv", ["x", "--steps", "256", "--seeds", "8", "--roles", "nnknn_nnknn",
                                      "--audit-queries-per-batch", "2", "--quality-tracking", "--output", str(tmp_path / "roles"),
                                      "--study", "src"])
    role_pilot.main()
    runs_json = tmp_path / "roles" / "manifest.json"
    manifest = json.loads(runs_json.read_text(encoding="utf-8"))
    runs_json.write_text(json.dumps({"runs": manifest["roles"]}), encoding="utf-8")

    start_rng = tr.capture_rng()

    def run(study, out):  # each call starts from the same global RNG state, as a fresh process would
        tr.restore_rng(start_rng)
        monkeypatch.setattr(sys, "argv", ["x", "--source", str(ROOT), "--runs", str(runs_json),
                                          "--output", str(tmp_path / out), "--study", study])
        pilot.main()
        return json.loads((tmp_path / out / "manifest.json").read_text(encoding="utf-8"))

    straight = run("straight", "a")
    assert len(straight["runs"]) >= 4
    real, calls = retention.constrained_retain, []

    def crashing(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise Crash("third condition")
        return real(*a, **k)

    monkeypatch.setattr(retention, "constrained_retain", crashing)
    with pytest.raises(Crash):
        run("partial", "b")
    counted = []
    monkeypatch.setattr(retention, "constrained_retain", lambda *a, **k: counted.append(1) or real(*a, **k))
    rerun = run("partial", "c")
    assert len(counted) == len(straight["runs"]) - 2  # the two complete conditions were not recomputed
    assert [r["condition"] for r in straight["runs"]] == [r["condition"] for r in rerun["runs"]]
    volatile = {"selection_seconds", "checkpoint"}
    for a, b in zip(straight["runs"], rerun["runs"]):
        ra, rb = tr.load_run(a["result"]), tr.load_run(b["result"])
        assert ({k: v for k, v in ra.record("result").items() if k not in volatile}
                == {k: v for k, v in rb.record("result").items() if k not in volatile}), a["condition"]
        sa, sb = states_of(a["result"]), states_of(b["result"])
        assert sa == sb, (a["condition"], [k for k in sa if sa[k] != sb.get(k)])


def test_harvest_p17_reads_only_run_units(root, tmp_path):
    for cap in (16, 32):
        def cfg(cap=cap):
            return make_nnknn_rl_config("smoke", seed=8, total_timesteps=200, case_capacity=cap)
        record(f"cap{cap}", train_nnknn_rl, "cartpole", cfg, study="p17",
               config=t2runs.unit_config("nnknn_rl", "cartpole", cfg().to_dict(), "cpu"))
    legacy = root / "t2" / "p17" / "legacy_run_dir"
    legacy.mkdir()
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "manifest.json"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "harvest_p17.py"), str(root / "t2" / "p17"), str(out)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(r["capacity"] for r in manifest["runs"]) == [16, 32]
    for r in manifest["runs"]:
        assert r["seconds"] == t2runs.RunFiles(r["run_dir"]).train_seconds()
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "harvest_p17.py")], capture_output=True, text=True)
    assert proc.returncode != 0 and "RUNS_FOLDER" in proc.stderr


def test_resume_cadence_follows_the_task(root, monkeypatch):
    """Image (ALE) tasks checkpoint every IMAGE_RESUME_EVERY_SECONDS (their checkpoints hold the replay buffer),
    flat tasks at the core default; an explicit value and NNKNN_RESUME_EVERY_SECONDS still win."""
    assert t2runs.resume_every_seconds("cartpole") is None
    assert t2runs.resume_every_seconds("pong") == t2runs.IMAGE_RESUME_EVERY_SECONDS == 1800.0
    assert t2runs.resume_every_seconds("no-such-task") is None and t2runs.resume_every_seconds(None) is None
    monkeypatch.delenv("NNKNN_RESUME_EVERY_SECONDS")
    seen = {}
    for name, task, every in (("flat", "cartpole", "auto"), ("image", "pong", "auto"), ("given", "pong", 42.0)):
        with t2runs.study_unit("cadence", (name,), t2runs.unit_config("dqn", task, {}, "cpu"),
                               checkpoint_every_seconds=every) as (source, run):
            assert source == "fresh"
            seen[name] = run._every_seconds
            run.load_resume()
    assert seen == {"flat": 300.0, "image": 1800.0, "given": 42.0}
    monkeypatch.setenv("NNKNN_RESUME_EVERY_SECONDS", "7")
    with t2runs.study_unit("cadence", ("env",), t2runs.unit_config("dqn", "pong", {}, "cpu")) as (_, run):
        assert run._every_seconds == 7.0
        run.load_resume()


@pytest.mark.parametrize("name", ["dqn", "ppo", "nnknn"])
def test_train_seconds_add_up_over_sessions(root, monkeypatch, name):
    """x/sys/train_seconds sums the sessions: with a clock that ticks once per reading, an interrupted and resumed
    unit reports what the uninterrupted one does (the killed session counts up to the checkpoint resumed from)."""
    ticks = iter(range(1, 10 ** 6))
    monkeypatch.setattr(t2runs, "time", types.SimpleNamespace(perf_counter=lambda: float(next(ticks))))
    train, task, cfg = CASES[name]
    whole, _, _ = record(name, train, task, cfg, study="whole")
    resumed, _, was_resumed = record(name, train, task, cfg, study="cut", crash_after=2)
    assert was_resumed
    n_ckpt = len(tr.load_run(whole).sessions()[0]["checkpoints"])
    assert n_ckpt > 2
    assert t2runs.RunFiles(whole).train_seconds() == float(n_ckpt + 1)  # one tick per checkpoint and at finish
    assert t2runs.RunFiles(resumed).train_seconds() == t2runs.RunFiles(whole).train_seconds()
    assert len(tr.load_run(resumed).series("sys/train_seconds")) == 1
