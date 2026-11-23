"""t1_parallel: every job records into its study (--study is required), and crash-rerun logs are kept."""
import contextlib
import io
import json
import subprocess
import sys

import pytest

import tools.t1_parallel as parallel


def test_parallel_and_execute_study_require_a_study(tmp_path, monkeypatch):
    import tools.t1_execute_study as execute
    for module, argv in ((parallel, ["t1_parallel.py", "configs/t1/smoke_mcb.yaml", "--out", str(tmp_path), "--logs",
                                     str(tmp_path)]),
                         (execute, ["t1_execute_study.py", str(tmp_path), "--out", str(tmp_path)])):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc, contextlib.redirect_stderr(io.StringIO()):
            module.main()
        assert exc.value.code == 2


def test_rerun_keeps_earlier_attempt_logs(tmp_path, monkeypatch):
    attempts = iter([(1, "attempt 1"), (1, "attempt 2"), (0, "attempt 3")])

    def fake_run(cmd, *, stdout, **kwargs):
        assert cmd[-2:] == ["--study", "s"]  # every job records its runs as units
        code, text = next(attempts)
        stdout.write(text + "\n")
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    out, logs = tmp_path / "out", tmp_path / "logs"
    logs.mkdir()
    for expected in (1, 1, 0):
        assert parallel.run_job(tmp_path / "toy.yaml", "toy", 3, out, logs, study="s")["exit_code"] == expected
    log = logs / "toy__s3.log"
    assert log.read_text(encoding="utf-8") == "attempt 3\n"
    assert (logs / "toy__s3.attempt1.log").read_text(encoding="utf-8") == "attempt 1\n"
    assert (logs / "toy__s3.attempt2.log").read_text(encoding="utf-8") == "attempt 2\n"
    assert json.loads((logs / "toy__s3.json").read_text())["exit_code"] == 0
    assert parallel.run_job(tmp_path / "toy.yaml", "toy", 3, out, logs, study="s")["reused"]  # a passed job is not rerun
    assert sorted(p.name for p in logs.iterdir()) == ["toy__s3.attempt1.log", "toy__s3.attempt2.log",
                                                       "toy__s3.json", "toy__s3.log"]
    (logs / "toy__s3.attempt1.log").unlink()
    parallel.keep_previous_log(log)  # first free N
    assert (logs / "toy__s3.attempt1.log").read_text(encoding="utf-8") == "attempt 3\n" and not log.exists()
