"""Test helpers: a recording T1 runner and read views of its run units (every T1 run is recorded; --study)."""
from __future__ import annotations

import fnmatch
from pathlib import Path

from model.common import tracking as tr
from model.t1 import records

STUDY = "t"


def study_runner(cfg, tmp_path, monkeypatch, study=STUDY):
    """A ``Runner`` recording into ``<tmp_path>/runs`` (``NNKNN_RUNS``); its input snapshots go to ``<tmp_path>/inputs``."""
    from tools.t1_run import Runner
    runs = Path(tmp_path) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    for name in ("NNKNN_RESUME", "NNKNN_RESUME_EVERY_SECONDS", "NNKNN_RESUME_EVERY_STEPS"):
        monkeypatch.delenv(name, raising=False)
    return Runner(cfg, Path(tmp_path) / "inputs", study)


class Unit:
    """A complete condition unit read like the per-run files the runner wrote before recording was mandatory."""

    RECORD = {"run_manifest.json": "manifest", "metrics.json": "metrics", "history.json": "history"}

    def __init__(self, row, data):
        self.row, self.data, self.path = row, data, Path(data.path)
        self.condition, self.seed = row["condition"], row["seed"]

    def json(self, name):
        return records.record(self.data, self.RECORD.get(name, name.removesuffix(".json")))

    def jsonl(self, name):
        return records.rows(self.data, name.removesuffix(".jsonl"))

    def torch(self, name):
        return self.data.load_state(name.removesuffix(".pt"))

    def states(self):
        return sorted(self.data.states())


def units(root, condition="*", dataset="*"):
    """Every complete condition unit below ``root`` (a run root or a study/experiment tree), in shard order."""
    return [Unit(row, data) for row, data in records.condition_units(Path(root))
            if fnmatch.fnmatchcase(row["condition"], condition) and fnmatch.fnmatchcase(row["dataset"], dataset)]


def unit(root, condition="*", dataset="*"):
    found = units(root, condition, dataset)
    assert found, f"no complete unit {dataset}/{condition} below {root}"
    return found[0]


def rows(root):
    return [u.row for u in units(root)]


def stage_units(root):
    """Paths of the complete stage units (``_<stage>``) below ``root``."""
    return sorted(Path(i.path) for i in tr.iter_runs(Path(root)) if Path(i.path).parent.name.startswith("_"))


def units_from_files(src, runs, study="fx", experiment=None):
    """Record a synthetic experiment folder written by a test in the old file layout (``runs__*.jsonl`` shard rows,
    ``<dataset>/<condition>/s<seed>/`` with ``run_manifest.json``, ``metrics.json``, ``history.json`` and the
    ``*.jsonl`` logs) as the condition units ``t1/<study>/<experiment>/...`` below the run root ``runs``; returns that
    unit tree. Only test fixtures are converted this way: the tools read run units only.

    Run directories without a shard row get the row ``{dataset, condition, seed}`` of their path. Units are written in
    shard order, so ``records.condition_units`` gives the rows back in that order.
    """
    import json

    src, runs = Path(src), Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    experiment = experiment or src.name
    rows = [json.loads(line) for f in sorted(src.glob("runs__*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {(r["dataset"], r["condition"], r["seed"]) for r in rows}
    for man in sorted(src.glob("*/*/s*/run_manifest.json")):
        d = man.parent
        key = (d.parent.parent.name, d.parent.name, int(d.name[1:]))
        if key not in seen:
            rows.append({"dataset": key[0], "condition": key[1], "seed": key[2]})
    for i, row in enumerate(rows):
        d = src / str(row["dataset"]) / str(row["condition"]) / f"s{row['seed']}"
        key = records.key(study, experiment, row["dataset"], row["condition"], f"s{row['seed']}")
        if tr.run_status(tr.unit_dir(key, runs)).status != "empty":  # a duplicate row: its own sibling unit
            key = records.key(study, experiment, row["dataset"], f"{row['condition']}.dup{i}", f"s{row['seed']}")
        with tr.open_run(key, config={"fixture": i, "row": row}, root=runs) as run:
            run.log_record("row", records.ordered(row))
            for name, record in (("run_manifest.json", "manifest"), ("metrics.json", "metrics"),
                                 ("history.json", "history")):
                if (d / name).is_file():
                    run.log_record(record, records.ordered(json.loads((d / name).read_text(encoding="utf-8"))))
            for name in ("case_statistics", "case_maintenance", "retrieval_events"):
                if (d / f"{name}.jsonl").is_file():
                    lines = (d / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                    run.log_records(name, [records.ordered(json.loads(line)) for line in lines if line.strip()])
    return records.unit_tree(study, experiment, root=runs)


def trial_units_from_files(grid, runs, study="grid"):
    """Record the ``test_predictions.pt`` of a synthetic grid or evaluate output (test fixtures in the old layout) as
    its grid trial units and name the study in its ``study.json``/``evaluate.json``, as a recording grid does.
    Trials already recorded are kept, so the conversion may be repeated."""
    import json

    import numpy as np
    import torch

    grid, runs = Path(grid), Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    for path in sorted(grid.glob("*/*/*/*/s*/test_predictions.pt")):
        seed_dir = path.parent
        cid, stage, family, task = seed_dir.parent.name, seed_dir.parent.parent.name, \
            seed_dir.parent.parent.parent.name, seed_dir.parent.parent.parent.parent.name
        key = records.key(study, task, family, stage, cid, seed_dir.name)
        if tr.run_status(tr.unit_dir(key, runs)).status == "complete":
            continue
        saved = torch.load(path, weights_only=True)
        with tr.open_run(key, config={"fixture": str(path.relative_to(grid))}, root=runs) as run:
            run.log_array("test_prediction", np.asarray(saved["prediction"].numpy()))
            run.log_array("test_target", np.asarray(saved["target"].numpy()))
    names = [n for n in ("study.json", "evaluate.json") if (grid / n).is_file()] or ["study.json"]
    for name in names:
        record = json.loads((grid / name).read_text(encoding="utf-8")) if (grid / name).is_file() else {}
        (grid / name).write_text(json.dumps({**record, "study": study}), encoding="utf-8")
