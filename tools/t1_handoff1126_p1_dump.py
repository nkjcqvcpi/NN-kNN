"""Read-only dump of the run units behind handoff pack p1 (2026-11-26) for its summary.

    python tools/t1_handoff1126_p1_dump.py --root <NN-KNN/runs> --out <dump.jsonl> [--part allcaseF|baselines|plan]... [--workers N]

Reads only run units under ``<root>/t1`` (TensorBoard event files, through ``model.common.tracking``); writes one JSON
line per unit and touches nothing under the run root. Parts:

- ``allcaseF``: every unit of ``handoff_allcaseF_r760/{allcaseF,maskedF}_20261126`` (seeds 50-52).
- ``baselines``: the grid evaluation trial units of ``handoff_allcaseF_r760/<task>/<family>/evaluation`` and, for the
  seven tasks whose matched baselines were recorded earlier, ``baselines_r760/<task>/<family>/evaluation`` (seeds 50-52).
- ``plan``: every unit of ``handoff_allcaseplan_r760`` and the seed 30-32 units of ``plan_full_r760``.

Each line: ``kind`` (row | trial | unit), ``study``, ``experiment``, ``rel`` (unit path below ``t1``), ``status``,
``host`` (from the event-file name), and for complete condition units the scalar fields of the ``row`` record, for
complete trial units the ``trial`` record's task key, family, seed and test metrics. Stage units are listed with their
status only. Exploratory tooling for one summary; it changes no default and no other tool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.common.tracking import load_run, run_status  # noqa: E402
from model.t1.records import unordered  # noqa: E402

EMBARGO = ("sel_fresh", "held_out")
EVAL50 = ("classification__iris", "classification__wine", "classification__zebra", "classification__zebra_special",
          "regression__body_fat", "regression__diabets", "regression__yacht")
FAMILIES = ("knn", "mlp", "catboost", "tabm")


def units(base: Path, seeds: set[int] | None):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "weights" and not any(w in d.lower() for w in EMBARGO))
        ev = sorted(f for f in filenames if "tfevents" in f)
        if ev:
            dirnames[:] = []
            leaf = Path(dirpath).name
            if seeds is not None and not (leaf.startswith("s") and leaf[1:].isdigit() and int(leaf[1:]) in seeds):
                continue
            yield Path(dirpath), ev


def scalars(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) or v is None}


def host_of(ev: list[str]) -> str:
    parts = ev[-1].split(".")
    return parts[4] if len(parts) > 4 else ""


def one(job: tuple) -> str:
    t1, study, exp, path, ev = job
    try:
        data = load_run(path, arrays=False, allow_incomplete=True)
        info = data.info
    except Exception:  # invalid: status and reason only
        data, info = None, run_status(path)
    line = {"kind": "unit", "study": study, "experiment": exp, "rel": str(path.relative_to(t1)), "status": info.status,
            "reason": info.reason, "host": host_of(ev), "n_event_files": len(ev)}
    if info.status == "complete" and data is not None:
        try:
            line.update(kind="row", row=scalars(unordered(data.record("row"))))
        except KeyError:
            try:
                tr = unordered(data.record("trial"))
                line.update(kind="trial", trial={k: tr.get(k) for k in ("seed", "status", "stage", "candidate_id")},
                            test_metrics=scalars(dict(tr.get("test_metrics") or {})),
                            n_test=(tr.get("dataset") or {}).get("n_test"))
            except KeyError:
                pass
    return json.dumps(line, default=str)


def jobs_of(t1: Path, study: str, sub: str, seeds: set[int] | None) -> list[tuple]:
    return [(t1, study, sub.split("/")[0], path, ev) for path, ev in units(t1 / study / sub, seeds)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="run root (NN-KNN/runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", action="append", choices=["allcaseF", "baselines", "plan"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t1 = Path(a.root) / "t1"
    jobs = []
    if "allcaseF" in a.part:
        for exp in ("allcaseF_20261126", "maskedF_20261126"):
            jobs += jobs_of(t1, "handoff_allcaseF_r760", exp, {50, 51, 52})
    if "baselines" in a.part:
        st = t1 / "handoff_allcaseF_r760"
        for task in sorted(p.name for p in st.iterdir() if p.name.startswith(("classification__", "regression__"))):
            for fam in FAMILIES:
                if (st / task / fam / "evaluation").is_dir():
                    jobs += jobs_of(t1, "handoff_allcaseF_r760", f"{task}/{fam}/evaluation", {50, 51, 52})
        for task in EVAL50:
            for fam in FAMILIES:
                if (t1 / "baselines_r760" / task / fam / "evaluation").is_dir():
                    jobs += jobs_of(t1, "baselines_r760", f"{task}/{fam}/evaluation", {50, 51, 52})
    if "plan" in a.part:
        for exp in sorted(p.name for p in (t1 / "handoff_allcaseplan_r760").iterdir() if p.is_dir()):
            jobs += jobs_of(t1, "handoff_allcaseplan_r760", exp, {30, 31, 32})
            if (t1 / "plan_full_r760" / exp).is_dir():
                jobs += jobs_of(t1, "plan_full_r760", exp, {30, 31, 32})
    # biggest event files first, so the long ones do not finish last
    jobs.sort(key=lambda j: -sum((j[3] / e).stat().st_size for e in j[4]))
    print(f"{len(jobs)} units", file=sys.stderr)
    with open(a.out, "w") as out, Pool(a.workers) as pool:
        for line in pool.imap_unordered(one, jobs):
            out.write(line + "\n")
            out.flush()


if __name__ == "__main__":
    main()
