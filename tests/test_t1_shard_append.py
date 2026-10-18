"""Shard appends from concurrent t1_run jobs, and t1_parallel crash-rerun logs."""
import json
import multiprocessing
import subprocess
from pathlib import Path

import tools.t1_parallel as parallel
from tools.t1_run import append_row

WORKERS, ROWS = 8, 50


def _append_rows(shard: str, worker: int, barrier) -> None:
    barrier.wait()  # all workers append at once
    for i in range(ROWS):
        append_row(Path(shard), {"run_id": f"w{worker}-r{i}", "worker": worker, "row": i, "pad": "x" * 2000})


def test_concurrent_appends_keep_every_row_once(tmp_path):
    shard = tmp_path / "runs__toy.jsonl"
    ctx = multiprocessing.get_context("spawn")  # Windows' start method, where seek-then-write lost rows
    barrier = ctx.Barrier(WORKERS)
    procs = [ctx.Process(target=_append_rows, args=(str(shard), w, barrier)) for w in range(WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=150)
    for p in procs:  # a deadlocked worker must fail the test, not hang interpreter exit
        if p.is_alive():
            p.terminate()
            p.join()
    assert [p.exitcode for p in procs] == [0] * WORKERS
    lines = shard.read_bytes().splitlines()
    rows = [json.loads(line) for line in lines]  # every line is valid JSON
    assert len(rows) == WORKERS * ROWS
    assert sorted(r["run_id"] for r in rows) == sorted(f"w{w}-r{i}" for w in range(WORKERS) for i in range(ROWS))
    assert all(len(r["pad"]) == 2000 for r in rows)
    assert sorted(tmp_path.glob("runs__*.jsonl")) == [shard]  # readers' glob skips the lock file


def test_append_bytes_match_the_text_mode_append(tmp_path):
    rows = [{"run_id": "e-iris-a-s0", "seed": 0, "condition": "café", "test_accuracy_pre": 0.9333333333333333,
             "val_loss": float("nan"), "ratio": float("inf"), "frozen": True, "temperature": None},
            {"zeta": 1, "alpha": -2.5e-07, "name": "é\t\"q\"\n"}]
    new, old = tmp_path / "runs__new.jsonl", tmp_path / "runs__old.jsonl"
    for row in rows:
        append_row(new, row)
        with old.open("a", encoding="utf-8") as fh:  # the append this helper replaced
            fh.write(json.dumps(row) + "\n")
    assert new.read_bytes() == old.read_bytes()
    assert [json.loads(line) for line in new.read_text(encoding="utf-8").splitlines()][1] == rows[1]


def test_rerun_keeps_earlier_attempt_logs(tmp_path, monkeypatch):
    attempts = iter([(1, "attempt 1"), (1, "attempt 2"), (0, "attempt 3")])

    def fake_run(cmd, *, stdout, **kwargs):
        code, text = next(attempts)
        stdout.write(text + "\n")
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    out, logs = tmp_path / "out", tmp_path / "logs"
    logs.mkdir()
    for expected in (1, 1, 0):
        assert parallel.run_job(tmp_path / "toy.yaml", "toy", 3, out, logs)["exit_code"] == expected
    log = logs / "toy__s3.log"
    assert log.read_text(encoding="utf-8") == "attempt 3\n"
    assert (logs / "toy__s3.attempt1.log").read_text(encoding="utf-8") == "attempt 1\n"
    assert (logs / "toy__s3.attempt2.log").read_text(encoding="utf-8") == "attempt 2\n"
    assert json.loads((logs / "toy__s3.json").read_text())["exit_code"] == 0
    assert parallel.run_job(tmp_path / "toy.yaml", "toy", 3, out, logs)["reused"]  # a passed job is not rerun
    assert sorted(p.name for p in logs.iterdir()) == ["toy__s3.attempt1.log", "toy__s3.attempt2.log",
                                                       "toy__s3.json", "toy__s3.log"]
    (logs / "toy__s3.attempt1.log").unlink()
    parallel.keep_previous_log(log)  # first free N
    assert (logs / "toy__s3.attempt1.log").read_text(encoding="utf-8") == "attempt 3\n" and not log.exists()
