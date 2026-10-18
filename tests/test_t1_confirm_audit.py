"""tools/t1_confirm_audit.py on genuine tools/t1_run.py output (tiny confirmation-config runs)."""
import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import torch

from model.t1.data import T1Data
from tools.t1_confirm_audit import (RebuildError, audit, done_line, done_lines, main, read_shard, rebuild_row, repair,
                                    row_line, shard_lock)
from tools.t1_run import Runner, load_config

EXP = "confirm_tiny"


def tiny_data(ds, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(16, 3, generator=g)
    if ds["task_type"] == "classification":
        y = (X[:, 0] > 0).long()
        return T1Data(ds["name"], "classification", X, y, X[:8] + .05, y[:8], X[8:] - .05, y[8:], num_classes=2)
    y = X[:, 0] + .1 * X[:, 1]
    return T1Data(ds["name"], "regression", X, y, X[:8] + .01, y[:8], X[8:] + .02, y[8:])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One experiment from a frozen confirmation config: wine in two roles (qualified keys) plus an
    unqualified task, two seeds, the confirmation's four condition types (8 rows per shard). Each
    (dataset, seed) job's stdout is kept as tools/t1_parallel.py keeps it: ``logs/<task>__s<seed>.log``
    beside a receipt with its exit code."""
    cfg = load_config(Path("configs/t1/confirm_final_20261013/regression__wine.yaml"))
    cfg["experiment"] = EXP
    cfg["core"].update(epochs=2, patience=2, batch_size=4, top_k=3, bias_init_k=3)
    cfg["core_by_dataset"]["regression__wine"].update(top_k=3, batch_size=4)
    cfg["reuse"].update(epochs=2, patience=2, hidden_dims=[8, 4])
    cfg["sync"].update(epochs=2, patience=2, block_epochs=1)
    cfg["conditions"][0]["adapter_warmup_epochs"] = 1
    cfg["datasets"] = [{"name": "wine", "task_type": "regression", "qualified": True},
                       {"name": "wine", "task_type": "classification"}, {"name": "tiny", "task_type": "regression"}]
    out = tmp_path_factory.mktemp("built")
    runner = Runner(cfg, out)
    runner.data_for = tiny_data
    (out / "logs").mkdir()
    for ds in cfg["datasets"]:
        for seed in (40, 41):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                runner.run_encoder_ablation(ds, seed)
            log = out / "logs" / f"{ds['task_type']}__{ds['name']}__s{seed}.log"
            log.write_text(stdout.getvalue(), encoding="utf-8")
            log.with_suffix(".json").write_text(json.dumps({"dataset": ds["name"], "seed": seed, "exit_code": 0}))
    runner.source_path.unlink()  # the source snapshot is not audited
    return out


@pytest.fixture
def root(built, tmp_path):
    """A copy of the experiment directory; its job logs are ``logs_of(root)``."""
    return Path(shutil.copytree(built, tmp_path / "out")) / EXP


def logs_of(root):
    return root.parent / "logs"


def lines_of(path):
    return path.read_bytes().splitlines(keepends=True)


def drop_line(path, index):
    lines = lines_of(path)
    dropped = lines.pop(index)
    path.write_bytes(b"".join(lines))
    return dropped


def make_old(root, seconds=7200):
    """Backdate every file so the settle window sees finished runs."""
    t = time.time() - seconds
    for p in root.rglob("*"):
        os.utime(p, (t, t))


def test_every_row_rebuilds_exactly(root, tmp_path):
    shards = sorted(p.name for p in root.glob("runs__*.jsonl"))
    assert shards == ["runs__classification__wine.jsonl", "runs__regression__wine.jsonl", "runs__tiny.jsonl"]
    for shard in root.glob("runs__*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            run_dir = root / row["dataset"] / row["condition"] / f"s{row['seed']}"
            assert row_line(rebuild_row(run_dir)) == line
    index = done_lines(logs_of(root))
    for run_dir in root.glob("*/*/s*"):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        run_id = rebuild_row(run_dir)["run_id"]
        assert [line for _, line in index[run_id]] == [done_line(run_id, metrics)]  # as Runner.finish printed it
    rep = audit(root, settle_seconds=0)
    assert rep["ok"] and rep["run_dirs"] == rep["rows"] == 24 and not rep["incomplete_run_dirs"]
    assert all(s["self_check"]["passed"] and s["self_check"]["exact"] == 8 for s in rep["shards"].values())
    assert not rep["missing_rows"] and not rep["orphan_rows"] and rep["duplicates"] == {"identical": [], "conflicting": []}
    assert set(rep["source_fingerprints"]) == {"regression__wine", "classification__wine", "regression__tiny"}
    assert read_shard(root / "runs__tiny.jsonl")["terminators"] == [os.linesep]  # t1_run's text-mode newline
    out = tmp_path / "audit.json"
    assert main([str(root), "--settle-seconds", "0", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_deleted_row_is_detected_and_repaired(root, tmp_path):
    shard = root / "runs__regression__wine.jsonl"
    original = lines_of(shard)
    dropped = drop_line(shard, 3)
    row = json.loads(dropped)
    rep = audit(root, settle_seconds=0)
    assert [m["run"] for m in rep["missing_rows"]] == [["regression__wine", row["condition"], row["seed"]]]
    assert not rep["ok"] and rep["shards"]["regression__wine"]["self_check"]["passed"]
    with pytest.raises(SystemExit):  # --repair needs the job logs
        main([str(root), "--repair", "--settle-seconds", "0"])
    out = tmp_path / "audit.json"
    assert main([str(root), "--repair", "--logs", str(logs_of(root)), "--settle-seconds", "0", "--json", str(out)]) == 0
    after = lines_of(shard)
    assert after == original[:3] + original[4:] + [dropped]  # appended byte-identical, terminator included
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [a["run_id"] for a in report["repair"]["appended"]] == [row["run_id"]]
    assert report["ok"] and report["before_repair"]["problems"] == ["1 run directories without a row"]


def test_repair_refuses_a_shard_whose_self_check_fails(root):
    bad, good = root / "runs__tiny.jsonl", root / "runs__classification__wine.jsonl"
    drop_line(bad, 0)
    lines = lines_of(bad)
    row = json.loads(lines[1])
    row["test_rmse_pre"] += 1e-9
    lines[1] = (json.dumps(row) + os.linesep).encode()
    bad.write_bytes(b"".join(lines))
    before = bad.read_bytes()
    drop_line(good, 5)
    rep = audit(root, settle_seconds=0, logs=logs_of(root))
    check = rep["shards"]["tiny"]["self_check"]
    assert (check["rows"], check["exact"], check["passed"]) == (7, 6, False)
    assert check["mismatches"][0]["reason"] == "fields differ: test_rmse_pre"
    done = repair(root, rep)
    assert [r["shard"] for r in done["refused"]] == ["runs__tiny.jsonl"]
    assert done["refused"][0]["reason"].startswith("self-check failed: 6/7")
    assert [a["shard"] for a in done["appended"]] == ["runs__classification__wine.jsonl"]
    assert bad.read_bytes() == before
    after = audit(root, settle_seconds=0)
    assert not after["ok"] and len(after["missing_rows"]) == 1 and after["shards"]["classification__wine"]["self_check"]["passed"]


def test_unverifiable_rows_block_repair(root):
    shard = root / "runs__tiny.jsonl"
    drop_line(shard, 0)
    victim = json.loads(lines_of(shard)[0])
    shutil.rmtree(root / "tiny" / victim["condition"] / f"s{victim['seed']}")
    with shard.open("ab") as fh:
        fh.write(b'847, "n_cases": 12}' + os.linesep.encode())  # the tail of an overwritten row
    rep = audit(root, settle_seconds=0)
    assert [o["reason"] for o in rep["orphan_rows"]] == ["no run directory"]
    assert len(rep["shards"]["tiny"]["malformed"]) == 1 and len(rep["missing_rows"]) == 1
    assert repair(root, rep)["refused"][0]["reason"].startswith("self-check failed: 6/8")


def test_conflicting_and_identical_duplicates(root):
    shard = root / "runs__regression__wine.jsonl"
    lines = lines_of(shard)
    row = json.loads(lines[2])
    row["test_rmse_pre"] *= 2
    shard.write_bytes(b"".join(lines) + lines[0] + (json.dumps(row) + os.linesep).encode())
    rep = audit(root, settle_seconds=0)
    first, third = json.loads(lines[0]), json.loads(lines[2])
    assert [d["run"] for d in rep["duplicates"]["identical"]] == [[first["dataset"], first["condition"], first["seed"]]]
    assert [d["run"] for d in rep["duplicates"]["conflicting"]] == [[third["dataset"], third["condition"], third["seed"]]]
    assert rep["shards"]["regression__wine"]["self_check"]["exact"] == 9  # 8 rows and the identical copy
    assert rep["problems"] == ["1 runs with conflicting duplicate rows",
                               "runs__regression__wine.jsonl: self-check 9/10 rows rebuild exactly"]
    shard.write_bytes(b"".join(lines) + lines[0])
    assert audit(root, settle_seconds=0)["ok"]
    assert audit(root, settle_seconds=0, fail_identical_duplicates=True)["problems"] == ["1 runs with identical duplicate rows"]


def test_source_fingerprint_expectations(root):
    fp = audit(root, settle_seconds=0)["source_fingerprints"]["regression__tiny"]["fingerprints"].popitem()[0]
    manifest = next((root / "tiny").rglob("run_manifest.json"))
    man = json.loads(manifest.read_text(encoding="utf-8"))
    man["run"]["source_fingerprint_sha256"] = "f" * 64  # not part of the row or configuration id
    manifest.write_text(json.dumps(man, indent=2), encoding="utf-8")
    rep = audit(root, settle_seconds=0, expect=[("*wine", fp), ("regression__*", fp[:12]), ("missing_task", fp)])
    wine, regression, none = rep["fingerprint_expectations"]
    assert wine["passed"] and wine["tasks"] == ["classification__wine", "regression__wine"]
    assert not regression["passed"] and regression["other_fingerprints"] == {"regression__tiny": {"f" * 64: 1}}
    assert not none["passed"] and none["tasks"] == []
    assert rep["source_fingerprints"]["regression__tiny"]["fingerprints"] == {fp: 7, "f" * 64: 1}
    assert all(s["self_check"]["passed"] for s in rep["shards"].values())
    assert main([str(root), "--settle-seconds", "0", "--expect-fingerprint", f"tiny={fp}"]) == 1
    assert main([str(root), "--settle-seconds", "0", "--expect-fingerprint", f"classification__*={fp}"]) == 0


def test_settle_window_and_lock_defer_repair(root):
    shard = root / "runs__tiny.jsonl"
    drop_line(shard, 0)
    rep = audit(root, settle_seconds=3600, logs=logs_of(root))  # runs just finished: a writer may still append
    assert len(rep["pending_rows"]) == 1 and not rep["missing_rows"] and rep["ok"]
    make_old(root)
    shard.write_bytes(shard.read_bytes())  # the shard itself changed just now
    rep = audit(root, settle_seconds=3600, logs=logs_of(root))
    assert len(rep["missing_rows"]) == 1
    assert "changed within 3600s" in repair(root, rep)["refused"][0]["reason"]
    make_old(root)
    rep = audit(root, settle_seconds=3600, logs=logs_of(root))
    with shard_lock(shard):
        done = repair(root, rep, lock_timeout=0.2)
    assert done["refused"][0]["reason"] == "runs__tiny.jsonl.lock stayed locked for 0s" and not done["appended"]
    assert repair(root, rep)["appended"] and audit(root, settle_seconds=0)["ok"]


def test_rebuild_rejects_a_manifest_from_another_condition(root):
    run_dir = next((root / "tiny").glob("tuned__F/s40"))
    man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    man["resolved_config"]["condition"]["w_post"] = 0.25
    (run_dir / "run_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(RebuildError):
        rebuild_row(run_dir)
    rep = audit(root, settle_seconds=0)
    assert [e["run_dir"] for e in rep["rebuild_errors"]] == ["tiny/tuned__F/s40"]
    assert rep["shards"]["tiny"]["self_check"]["mismatches"][0]["reason"].startswith("rebuild failed: RebuildError")


def test_repair_needs_proof_that_finish_completed(root):
    """Only a run whose job log holds its [done] line (receipt exit 0) is repaired; a crash after
    metrics.json leaves an incomplete directory, never a missing row, and fails once settled."""
    shard, logs = root / "runs__tiny.jsonl", logs_of(root)
    row = json.loads(drop_line(shard, 0))
    run_dir = root / "tiny" / row["condition"] / f"s{row['seed']}"
    log = logs / f"regression__tiny__s{row['seed']}.log"
    attempt, receipt = log.with_name(f"{log.stem}.attempt1.log"), log.with_suffix(".json")
    text, receipt_text = log.read_text(encoding="utf-8"), receipt.read_text()
    line = done_line(row["run_id"], json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
    before = shard.read_bytes()
    reason = lambda: repair(root, audit(root, settle_seconds=0, logs=logs))["refused"][0]["reason"]  # noqa: E731
    assert repair(root, audit(root, settle_seconds=0))["refused"][0]["reason"].startswith("audited without job logs")
    receipt.write_text(json.dumps({"exit_code": 1}))
    assert reason() == "no log with the [done] line has a receipt with exit code 0"
    receipt.write_text(receipt_text)
    log.write_text(text.replace(line, line.replace("=", "=1")), encoding="utf-8")  # other metrics printed
    assert reason() == f"[done] line disagrees with metrics.json in {log.name}"
    log.write_text(text.replace(line, ""), encoding="utf-8")
    assert reason().startswith(f"no [done] {row['run_id']} line in any job log")
    attempt.write_text(text, encoding="utf-8")  # only a kept attempt (no receipt) printed it
    assert reason() == "no log with the [done] line has a receipt with exit code 0"
    assert shard.read_bytes() == before
    attempt.unlink()
    log.write_text(text, encoding="utf-8")

    # killed while writing retrieval_events.jsonl: metrics.json is there, the row is not;
    # the other seed of the condition was killed right after its manifest
    for name in ("retrieval_events.jsonl", "checkpoint.pt"):
        (run_dir / name).unlink()
    other_seed = {40: 41, 41: 40}[row["seed"]]
    other = root / "tiny" / row["condition"] / f"s{other_seed}"
    drop_line(shard, next(i for i, x in enumerate(lines_of(shard))
                          if (json.loads(x)["condition"], json.loads(x)["seed"]) == (row["condition"], other_seed)))
    for p in other.iterdir():
        if p.name != "run_manifest.json":
            p.unlink()
    rep = audit(root, logs=logs, settle_seconds=3600)  # changed just now: in progress, not judged
    assert rep["ok"] and not rep["missing_rows"] and not rep["pending_rows"]
    assert [i["recent"] for i in rep["incomplete_run_dirs"]] == [True, True]
    rep = audit(root, logs=logs, settle_seconds=0)
    assert not rep["ok"] and not rep["missing_rows"]
    assert {(tuple(i["run"]), tuple(i["missing_files"]), i["row"]) for i in rep["incomplete_run_dirs"]} == {
        (("tiny", row["condition"], row["seed"]), ("retrieval_events.jsonl", "checkpoint.pt"), False),
        (("tiny", row["condition"], other_seed), ("metrics.json", "history.json", "case_statistics.jsonl",
                                                  "case_maintenance.jsonl", "retrieval_events.jsonl", "checkpoint.pt"), False)}
    assert rep["problems"] == ["2 run directories without every file finish() writes before the row, unchanged for 0s"
                               " (crashed and not rerun, or files removed)"]
    settled = shard.read_bytes()
    assert repair(root, rep) == {"appended": [], "refused": []} and shard.read_bytes() == settled
