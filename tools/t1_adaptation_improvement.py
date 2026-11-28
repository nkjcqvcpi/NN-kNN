"""Local paired comparison of existing and magnitude/direction adaptation.

The frozen 2026-10-13 task configurations define the 29-task suite and its F
schedule (frozen reuse for diabetes). These are NEW local pairs, not a rerun or
replacement of r760's historical numbers. Each pair shares one recorded core,
data, and starting RNG state. All complete units are reused, never retrained.

Example: python tools/t1_adaptation_improvement.py --study adaptation_v1
         --tasks classification__iris regression__yacht --mode run
Use --mode preflight to check local datasets, or --mode summarize to read only
recorded run units. Missing datasets abort before any selected job is trained.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import torch

from model.common import tracking
from model.t1 import records
from tools.t1_run import Runner, arm_fields, arm_label, load_config, near_scale_of

EXPERIMENT = "adaptation_improvement_v1"
SEEDS = (40, 41, 42)
EXCLUDED = {"regression__tabarena_363697"}
DEFAULT_CONFIGS = REPO / "configs" / "t1" / "confirm_final_20261013"
ARMS = ("existing", "magnitude_direction")


def comparison_config(source: dict, *, initial_scale: float = 0.01) -> dict:
    """Keep F's loss coefficients/budget/schedule; change only adapter design."""
    cfg = copy.deepcopy(source)
    if len(cfg["datasets"]) != 1 or cfg["arms"] != [{"name": "tuned"}]:
        raise ValueError("comparison requires one dataset and the frozen tuned arm")
    ds = cfg["datasets"][0]
    final = [c for c in cfg.get("conditions", []) if c.get("name") == "F"]
    frozen = not final and ds["name"] == "diabets" and ds["task_type"] == "regression"
    if len(final) != 1 and not frozen:
        raise ValueError(f"no unique frozen F condition for {ds}")
    if frozen:
        if len(cfg.get("reuse_conditions", [])) != 1:
            raise ValueError("diabetes requires its unique frozen reuse condition")
        original = cfg["reuse_conditions"][0]
    else:
        original = final[0]
    old = {**original, "name": "existing"}
    new = {**original, "name": "magnitude_direction", "adapter_design": "magnitude_direction",
           "initial_scale": initial_scale}
    if frozen:
        cfg["reuse_conditions"], cfg["conditions"] = [old, new], []
    else:
        cfg["reuse_conditions"], cfg["conditions"] = [], [old, new]
    cfg["experiment"], cfg["seeds"] = EXPERIMENT, list(SEEDS)
    cfg["status"] = "exploratory_local_paired_comparison_seen_seeds"
    cfg["adapter_comparison"] = {
        "protocol": "local_shared_core_v1", "reference_experiment": source["experiment"],
        "reference_condition": copy.deepcopy(original), "matched_v_small": True,
        "initial_scale": initial_scale, "frozen_reuse": frozen,
        "historical_report": "reports/NEURAL_CBR_WEEKLY_2026-11-24.md#2",
        "interpretation": "new local paired runs; not matched r760 historical observations",
    }
    return cfg


def suite(config_dir: Path = DEFAULT_CONFIGS, tasks=None, *, initial_scale=0.01) -> dict[str, dict]:
    files = {p.stem: p for p in sorted(config_dir.glob("*.yaml")) if p.stem not in EXCLUDED}
    selected = sorted(files if tasks is None else set(tasks))
    if not selected or set(selected) - files.keys():
        raise ValueError(f"unknown or empty task selection: {sorted(set(selected) - files.keys())}")
    return {name: comparison_config(load_config(files[name]), initial_scale=initial_scale) for name in selected}


def data_fingerprint(data) -> str:
    tensors = {name: getattr(data, name) for name in (
        "X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "X_maint", "y_maint")}
    return tracking.state_digest({"tensors": tensors, "description": records.jsonable(data.describe()),
        "y_scale": data.y_scale, "nominal": records.jsonable({name: getattr(data, name) for name in
        ("nominal_train", "nominal_val", "nominal_test", "nominal_maint")})})


def preflight(configs: dict[str, dict], seeds=SEEDS) -> dict[str, dict[int, str]]:
    """Load every selected dataset first, without creating a runner or training."""
    digests, errors = {}, []
    for task, cfg in configs.items():
        probe = object.__new__(Runner)
        probe.cfg = cfg
        digests[task] = {}
        for seed in seeds:
            try:
                data = probe.data_for(cfg["datasets"][0], int(seed))
                digests[task][int(seed)] = data_fingerprint(data)
                print(f"[preflight] {task} s{seed}: {len(data.y_train)} train / "
                      f"{len(data.y_val)} validation / {len(data.y_test)} test", flush=True)
            except Exception as exc:
                errors.append(f"{task} s{seed}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("Dataset preflight failed; no training was started. Select available tasks with --tasks.\n"
                         + "\n".join(errors))
    return digests


class ComparisonRunner(Runner):
    """Existing training paths, one core per seed, paired adapter initial RNG."""

    def __init__(self, cfg, out_root, study, fingerprints):
        self.fingerprints = fingerprints
        super().__init__(cfg, out_root, study)

    def run_config(self, ds, seed, cond):
        return {**super().run_config(ds, seed, cond), "data_fingerprint": self.fingerprints[int(seed)]}

    def data_for(self, ds, seed):
        data = super().data_for(ds, seed)
        if data_fingerprint(data) != self.fingerprints[int(seed)]:
            raise ValueError("data changed after preflight")
        return data

    def pair_units(self, ds):
        arm = self.cfg["arms"][0]
        retrieval = (arm_label(arm, "retrieval_only"), {**arm_fields(arm), "setting": "retrieval_only"})
        frozen = self.cfg["adapter_comparison"]["frozen_reuse"]
        conditions = self.cfg["reuse_conditions"] if frozen else self.cfg["conditions"]
        unit = (lambda c: self.reuse_unit(ds["task_type"], c, arm)) if frozen else (lambda c: self.sync_unit(c, arm))
        return retrieval, [(c, unit(c)) for c in conditions]

    def run_pair(self, ds, seed):
        self.check_recorded(ds, seed)
        retrieval, adaptations = self.pair_units(ds)
        units = [retrieval, *(u for _, u in adaptations)]
        if all(self.complete(ds, seed, *unit) is not None for unit in units):
            for unit in units:
                self.skip(ds, seed, *unit)
            return
        data, arm = self.data_for(ds, seed), self.cfg["arms"][0]
        core = self.core_stage(ds, seed, data, units, arm, observed=True)
        paired_rng = tracking.capture_rng()
        if not self.skip(ds, seed, *retrieval):
            self.retrieval_only(ds, seed, data, core["model"], core["cc"], core["tr"], core["tracker"], arm)
        for cond, unit in adaptations:
            if self.skip(ds, seed, *unit):
                continue
            tracking.restore_rng(paired_rng)
            # Deepcopy together keeps optimizer parameters bound to the copied core.
            base = copy.deepcopy({k: core[k] for k in ("model", "cc", "tr")})
            if self.cfg["adapter_comparison"]["frozen_reuse"]:
                self.reuse_condition(ds, seed, data, base["model"], base["cc"], base["tr"], cond, arm)
            else:
                self.sync_condition(ds, seed, data, base["model"], base["cc"], base["tr"], cond, arm,
                                    near_scale_of(base["model"]))


def check_recorded_suite(configs, fingerprints, seeds, study):
    """Reject incompatible completed units across the selected suite before training."""
    for task, cfg in configs.items():
        probe = object.__new__(ComparisonRunner)
        probe.cfg, probe.study, probe.exp = cfg, study, EXPERIMENT
        probe.fingerprints = fingerprints[task]
        for seed in seeds:
            probe.check_recorded(cfg["datasets"][0], seed)


def _error(row, output="post"):
    value = (1.0 - float(row[f"test_accuracy_{output}"]) if row["task_type"] == "classification"
             else float(row[f"test_rmse_{output}"]))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {output} error: {value}")
    return value


def aggregate_pairs(pairs: list[dict]) -> dict:
    """Separate task families; equal task weight, explicit zero-error exclusions."""
    tasks = []
    for task in sorted({r["task"] for r in pairs}):
        rows = [r for r in pairs if r["task"] == task]
        means = {k: float(np.mean([r[k] for r in rows])) for k in (
            "existing_error", "new_error", "retrieval_error", "existing_pre_error", "new_pre_error")}
        delta = means["new_error"] - means["existing_error"]
        tasks.append({"task": task, "task_type": rows[0]["task_type"], "n_seeds": len(rows), **means,
                      "delta": delta, "outcome": "tie" if abs(delta) <= 1e-12 else "win" if delta < 0 else "loss"})
    groups = {}
    for kind in ("classification", "regression"):
        rows = [r for r in tasks if r["task_type"] == kind]
        positive = [r for r in rows if r["new_error"] > 0 and r["existing_error"] > 0]
        groups[kind] = {"n_tasks": len(rows), "n_seed_pairs": sum(r["n_seeds"] for r in rows),
            "mean_task_delta": float(np.mean([r["delta"] for r in rows])) if rows else None,
            "wins": sum(r["outcome"] == "win" for r in rows),
            "ties": sum(r["outcome"] == "tie" for r in rows),
            "losses": sum(r["outcome"] == "loss" for r in rows),
            "gmr_positive_task_means": math.exp(float(np.mean([
                math.log(r["new_error"] / r["existing_error"]) for r in positive]))) if positive else None,
            "gmr_excluded_zero_error_tasks": [r["task"] for r in rows if r not in positive]}
    return {"tasks": tasks, "groups": groups}


def summarize(study: str, configs: dict, seeds=SEEDS) -> dict:
    """Read units only; absent units remain missing, never initiate training."""
    pairs, missing = [], []
    for task, cfg in configs.items():
        probe = object.__new__(ComparisonRunner)
        probe.cfg, probe.study, probe.exp = cfg, study, EXPERIMENT
        ds = cfg["datasets"][0]
        retrieval, adaptations = probe.pair_units(ds)
        for seed in seeds:
            loaded = {}
            for label, cond in [retrieval, *(u for _, u in adaptations)]:
                key = probe.unit_key(ds, label, seed)
                unit = tracking.load_unit(key, config=None, arrays=False)
                if unit is None:
                    missing.append(key.run_id)
                    continue
                recorded = unit.config or {}
                fingerprint = recorded.get("data_fingerprint")
                if not fingerprint:
                    raise ValueError(f"{key.run_id} has no data fingerprint")
                expected = Runner.unit_config({"experiment": EXPERIMENT, "dataset": ds, "seed": seed,
                    "condition": cond, "config": cfg, "data_fingerprint": fingerprint})
                if tracking.config_identity(recorded) != tracking.config_identity(expected):
                    raise tracking.ConfigMismatchError(f"{key.run_id} does not match this comparison protocol")
                loaded[label] = (records.record(unit, "row"), fingerprint, unit.info.run_id)
            if len(loaded) != 3:
                continue
            ro, old, new = [loaded[label] for label in (retrieval[0], adaptations[0][1][0], adaptations[1][1][0])]
            if len({ro[1], old[1], new[1]}) != 1:
                raise ValueError(f"unpaired data fingerprints for {task} seed {seed}")
            pairs.append({"task": task, "task_type": ds["task_type"], "seed": seed,
                "retrieval_error": _error(ro[0], "pre"), "existing_error": _error(old[0]),
                "new_error": _error(new[0]), "existing_pre_error": _error(old[0], "pre"),
                "new_pre_error": _error(new[0], "pre"), "delta": _error(new[0]) - _error(old[0]),
                "data_fingerprint": ro[1], "retrieval_unit": ro[2], "existing_unit": old[2], "new_unit": new[2]})
    return {"study": study, "protocol": "local_shared_core_v1", "exploratory": True,
        "interpretation": "new local paired evidence; historical r760 results are not pooled",
        "requested_tasks": len(configs), "reference_suite_tasks": 29, "requested_seeds": list(seeds),
        "expected_seed_pairs": len(configs) * len(seeds), "complete_seed_pairs": len(pairs),
        "missing_units": missing, "pairs": pairs, **aggregate_pairs(pairs),
        "limitations": ["Seeds 40–42 were previously seen; no confirmatory inference.",
            "Task win/tie/loss uses mean error and numerical tolerance 1e-12, not statistical significance.",
            "GMR weights positive-error task means equally and lists zero-error exclusions.",
            "Classification and regression are summarized separately; no cross-family pooled score.",
            "Historic loss coefficients including v_small are preserved to isolate architecture."]}


def write_summary(report: dict, out: Path):
    out = out.resolve()
    if not out.is_relative_to((REPO / "results").resolve()):
        raise ValueError("generated summaries must be under this worktree's results/")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("pairs", "tasks"):
        rows = report[name]
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--mode", choices=("run", "preflight", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", help="fully qualified keys; default all 29 tasks")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--initial-scale", type=float, default=0.01)
    parser.add_argument("--out", type=Path, help="summary/input directory under worktree results/")
    args = parser.parse_args(argv)
    if not set(args.seeds) <= set(SEEDS) or len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be a non-repeated subset of 40 41 42")
    if not math.isfinite(args.initial_scale) or args.initial_scale <= 0:
        parser.error("initial scale must be finite and positive")
    records.check_study(args.study)
    configs = suite(tasks=args.tasks, initial_scale=args.initial_scale)
    out = args.out or REPO / "results" / "adaptation_improvement" / args.study
    if not out.resolve().is_relative_to((REPO / "results").resolve()):
        parser.error("--out must be under this worktree's results/")
    if args.mode != "summarize":
        fingerprints = preflight(configs, args.seeds)
        check_recorded_suite(configs, fingerprints, args.seeds, args.study)
        if args.mode == "preflight":
            return fingerprints
        if os.environ.get("NNKNN_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True)
        for task, cfg in configs.items():
            runner = ComparisonRunner(cfg, out / "inputs", args.study, fingerprints[task])
            for seed in args.seeds:
                runner.run_pair(cfg["datasets"][0], seed)
    report = summarize(args.study, configs, args.seeds)
    write_summary(report, out)
    print(json.dumps({"complete_seed_pairs": report["complete_seed_pairs"], "expected_seed_pairs":
        report["expected_seed_pairs"], "groups": report["groups"], "summary": str(out / "comparison.json")}, indent=2))
    return report


if __name__ == "__main__":
    main()
