"""File T1 results and logs by T1 handoff phase (T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md section 13).

Layout:  results/t1/<phase>/{pilot_20260929,pi20260920}/...   logs/t1/<phase>/...

    python scripts/reorganize_t1_results_20261010.py --tracked --apply   # git mv the tracked pilot shards
    python scripts/reorganize_t1_results_20261010.py                     # dry run: ignored local artifacts
    python scripts/reorganize_t1_results_20261010.py --apply             # move them
    python scripts/reorganize_t1_results_20261010.py --pipeline [--apply]  # also file results/t1_pi20260920

Without --apply nothing is changed. Local moves refuse to run while a T1 process is
running from this checkout. Every move is recorded in results/t1/PATH_MAP.json.
Run artifacts are moved unchanged; paths recorded inside them stay as written. Six
audit scripts inside results/t1_pi20260920 locate the repository with
Path(__file__).parents[3]; after --pipeline they must be run from their original depth.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P0, P1, P2 = "p0_reproduce_instrument", "p1_supervised_retention", "p2_classification_reuse"
P4, P5, P6 = "p4_component_sync", "p5_mcb_interaction", "p6_confirmation_revision"
PILOT = {"p0_legacy_reference": P0, "p1_retention_classification": P1, "p1_retention_regression": P1,
         "p1_retention_synthetic": P1, "p2_reuse": P2, "p4_sync": P4, "p5_mcb": P5, "t12_revise_synthetic": P6}
PIPELINE = {
    "smoke_retention": P0, "smoke_reuse": P0, "smoke_sync": P0, "smoke_mcb": P0, "smoke_revise": P0,
    "verification_20261002": P0,
    "alignment_candidates_small": P1, "alignment_digits": P1, "continuation_20261002": P1,
    "plan_audit_20261009": P4, "plan_audit_20261009_repaired": P4,
    "radius_iris_20261009": P4, "radius_full_20261009": P4,
    "grid_preflight_20261009": P6, "grid_full_20261009": P6, "plan_iris_20261009": P6, "plan_full_20261009": P6,
    "full_study_20261009": P6, "runner_preflight_20261009": P6,
}
CONTINUATION_20261004 = {
    "core_selection_probe_before": P0, "core_selection_probe_after": P0,
    "adapted_budget_grid": P1, "adapted_reference_guard": P1, "adapted_regression": P1,
    "adapted_retraining_pilot": P1, "alignment_nominal_retention": P1,
    "alignment_nominal_reuse": P2, "optimizer_checkpoint_reuse": P2,
    "alignment_nominal_sync": P4, "alignment_sync": P4, "alignment_sync_selection_fix": P4,
    "optimizer_checkpoint_sync": P4, "sync_capacity_match": P4, "sync_capacity_match_complete": P4,
    "sync_capacity_match_failed_no_csv": P4,
    "alignment_revision": P6, "alignment_revision_matched": P6, "alignment_synthetic_reliability": P6,
    "reviewer_probes": P6, "reviewer_probes_v2": P6,
}
PIPELINE_LOGS = {"plan_audit_": P4, "grid_preflight_": P6}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def running_t1_processes() -> list[str]:
    if os.name != "nt":
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    else:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
                             capture_output=True, text=True).stdout
    root = str(ROOT).lower().replace("\\", "/")
    hits = []
    for line in out.splitlines():
        low = line.lower().replace("\\", "/")
        if ("t1_" in low or "full_study" in low or "wait_for_radius" in low) and (root in low or "results/t1_pi20260920" in low):
            if "reorganize_t1_results" not in low:
                hits.append(line.strip()[:160])
    return hits


def tracked_plan() -> list[tuple[str, str]]:
    return [(f"results/t1/{exp}", f"results/t1/{phase}/pilot_20260929/{exp}") for exp, phase in PILOT.items()
            if (ROOT / "results/t1" / exp).is_dir()]


def local_plan(pipeline: bool) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for exp, phase in PILOT.items():  # ignored per-run dirs left behind after the tracked move
        old = ROOT / "results/t1" / exp
        if old.is_dir():
            for child in sorted(old.iterdir()):
                moves.append((child, ROOT / "results/t1" / phase / "pilot_20260929" / exp / child.name))
    logs = ROOT / "logs/t1"
    if logs.is_dir():
        for f in sorted(p for p in logs.iterdir() if p.is_file()):
            exp = f.name.split("__")[0]
            if exp in PILOT:
                moves.append((f, logs / PILOT[exp] / "pilot_20260929" / f.name))
            elif f.name == "queue_status.txt":
                moves.append((f, logs / "pilot_20260929_queue_status.txt"))
            else:
                raise SystemExit(f"unclassified log {rel(f)}")
    for old, new in (("results/t0_battery", "results/t1/historical_t0/t0_battery"),
                     ("results/t1_invalid_fresh_adam", "results/t1/invalid_fresh_adam")):
        if (ROOT / old).is_dir():
            moves.append((ROOT / old, ROOT / new))
    if pipeline:
        pi = ROOT / "results/t1_pi20260920"
        for entry in sorted(pi.iterdir()):
            name = entry.name
            if entry.is_file():
                phase = next((ph for prefix, ph in PIPELINE_LOGS.items() if name.startswith(prefix)), None)
                if phase is None:
                    raise SystemExit(f"unclassified pipeline file {rel(entry)}")
                moves.append((entry, ROOT / "logs/t1" / phase / "pi20260920" / name))
            elif name == "continuation_20261004":
                for exp in sorted(entry.iterdir()):
                    if exp.name not in CONTINUATION_20261004:
                        raise SystemExit(f"unclassified experiment {rel(exp)}")
                    phase = CONTINUATION_20261004[exp.name]
                    moves.append((exp, ROOT / "results/t1" / phase / "pi20260920" / name / exp.name))
            elif name.startswith("pytest_"):
                moves.append((entry, ROOT / "results/t1" / P0 / "pi20260920" / "pytest" / name))
            elif name in PIPELINE:
                moves.append((entry, ROOT / "results/t1" / PIPELINE[name] / "pi20260920" / name))
            elif name == "__pycache__":
                continue
            else:
                raise SystemExit(f"unclassified pipeline entry {rel(entry)}")
    return moves


def record(entries: list[dict]) -> None:
    path = ROOT / "results/t1/PATH_MAP.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "description": "Old -> new locations after filing T1 results and logs by T1 handoff phase (2026-10-10). "
        "Artifacts were moved unchanged; paths recorded inside them are the paths at run time.",
        "moves": []}
    data["moves"].extend(entries)
    path.write_bytes((json.dumps(data, indent=1) + "\n").encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracked", action="store_true", help="git mv the tracked pilot experiment directories")
    ap.add_argument("--pipeline", action="store_true", help="also file results/t1_pi20260920 by phase")
    ap.add_argument("--apply", action="store_true", help="perform the moves (default: dry run)")
    ap.add_argument("--root", type=Path, help="checkout to plan for (default: this script's checkout)")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = args.root.resolve()
    if args.tracked:
        plan = tracked_plan()
        for old, new in plan:
            print(f"git mv {old} -> {new}")
        if args.apply:
            for old, new in plan:
                (ROOT / new).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
            record([{"old": o, "new": n, "kind": "tracked"} for o, n in plan])
        return 0
    if args.apply and (busy := running_t1_processes()):
        print("refusing to move files while T1 processes are running:", *busy, sep="\n  ")
        return 2
    plan = local_plan(args.pipeline)
    clashes = [rel(new) for _, new in plan if new.exists()]
    for old, new in plan:
        print(f"{rel(old)} -> {rel(new)}")
    print(f"{len(plan)} moves, {len(clashes)} destination clashes")
    if clashes:
        print(*clashes[:20], sep="\n  ")
        return 1
    if args.apply:
        for old, new in plan:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
        for exp in PILOT:
            old = ROOT / "results/t1" / exp
            if old.is_dir() and not any(old.iterdir()):
                old.rmdir()
        if args.pipeline and (ROOT / "results/t1_pi20260920/continuation_20261004").is_dir():
            (ROOT / "results/t1_pi20260920/continuation_20261004").rmdir()
        record([{"old": rel(o), "new": rel(n), "kind": "local"} for o, n in plan])
        print("applied; commit results/t1/PATH_MAP.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
