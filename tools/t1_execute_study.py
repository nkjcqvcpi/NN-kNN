"""Run the required T1 experiment families after validation-only HPO selection.

Full-capacity removal is intentionally separated from exact small-memory removal
diagnostics; neither is mislabeled as a scalable full-data result.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.common.artifacts import write_json, config_id, source_fingerprint
from tools.t1_grid_search import nnknn_core_settings
from tools.t1_run import load_config


def execute(config, dataset, seed, out):
    log = out / "logs" / f"{config.stem}__{dataset['task_type']}__{dataset['name']}__s{seed}.log"
    receipt = log.with_suffix(".json")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["exit_code"] == 0:
            return previous
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        run = subprocess.run([sys.executable, str(REPO / "tools/t1_run.py"), str(config),
            "--only-dataset", dataset["name"], "--seeds", str(seed), "--out", str(out / "experiments")],
            cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    result = dict(config=str(config), dataset=dataset, seed=seed, exit_code=run.returncode, log=str(log))
    write_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_output")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--only-dataset", nargs="*")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    grid = Path(args.grid_output)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((grid / "study.json").read_text())
    study = frozen["configuration"]
    if args.only_dataset:
        study["datasets"] = [d for d in study["datasets"] if d["name"] in args.only_dataset]
    selected = {}
    for ds in study["datasets"]:
        key = f"{ds['task_type']}__{ds['name']}"
        path = grid / key / "nnknn/selection.json"
        if not path.exists():
            raise ValueError(f"No validation-selected core for {key}; experiments cannot silently substitute defaults")
        selection = json.loads(path.read_text())
        if selection["test_stream_used"]:
            raise ValueError("Test-based HPO is prohibited")
        params = selection["parameters"]
        selected[key] = nnknn_core_settings(params)  # same settings the grid selected under
    classification = [d for d in study["datasets"] if d["task_type"] == "classification"]
    regression = [d for d in study["datasets"] if d["task_type"] == "regression"]
    # Core experiments evaluate real full datasets; no max_train cap here.
    # RRR and exact pair calibration costs are retained in elapsed-time records.
    groups = []
    def group(name, base, datasets, updates=None, tuned=True):
        cfg = load_config(REPO / "configs/t1" / base)
        cfg.update(experiment=name, status="exploratory_full_budget_plan_experiment", seeds=study["evaluation_seeds"], datasets=datasets)
        cfg["core"].update(epochs=300, patience=40)
        if tuned:
            cfg["core_by_dataset"] = selected
        for key, value in (updates or {}).items():
            cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
        # Separate task names prevent classification/regression wine collisions
        # in per-dataset subprocess selection.
        for ds in datasets:
            single = dict(cfg, datasets=[ds])
            filename = out / "configs" / f"{name}__{ds['task_type']}__{ds['name']}.yaml"
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text(yaml.safe_dump(single, sort_keys=False), encoding="utf-8")
            # Each task role has its own experiment namespace.
            single["experiment"] = f"{name}__{ds['task_type']}"
            filename.write_text(yaml.safe_dump(single, sort_keys=False), encoding="utf-8")
            groups.extend((name, filename, ds, seed) for seed in cfg["seeds"])
    standalone = [dict(name="synthetic_diag", task_type="classification", synthetic={"generator_version": "independent_v1"})]
    simple_policies = [dict(policy=p) for p in ("full_memory", "random", "stratified", "bias_current", "provenance_only", "bias_normalized", "kcenter", "coverage_reachability")]
    group("retention", "p1_retention_classification.yaml", classification + regression + standalone,
          dict(K=[.2, .5, .75], conditions=simple_policies))
    group("reuse_classification", "p2_reuse.yaml", classification + standalone)
    group("reuse_regression", "p2_reuse.yaml", regression,
          dict(conditions=[dict(output_mode="nominal_residual_scores", loss=l) for l in ("combined", "diff_only", "cls_only")]))
    group("synchronization", "full_sync_ablation_20261009.yaml", classification + regression + standalone)
    # The controlled encoder/MCB comparison requires the same representation
    # in both arms; raw/MLP HPO choices cannot confound this factor.
    group("sync_encoder_control", "sync_encoder_control_20261009.yaml", classification + regression + standalone, tuned=False)
    group("sync_mcb", "sync_mcb_interaction_20261009.yaml", classification + regression + standalone, tuned=False)
    group("mcb_retention", "p5_mcb.yaml", classification + regression + standalone, tuned=False)
    small = [dict(d, max_train=60) for d in classification + regression]
    group("exact_removal_diagnostics", "p1_retention_classification.yaml", small,
          dict(K=[.5, .75], conditions=[dict(policy=p) for p in ("random", "bias_normalized", "removal_influence", "removal_cached")]))
    group("removal_retraining_diagnostics", "alignment_retraining.yaml", [dict(d, max_train=24) for d in classification + regression])
    group("adapted_retraining_diagnostics", "alignment_adapted_retraining.yaml", [dict(d, max_train=18) for d in classification + regression])
    group("nominal_reuse", "alignment_nominal_reuse.yaml", [dict(name="synthetic_nominal", task_type="classification")], tuned=False)
    group("nominal_sync", "alignment_nominal_sync.yaml", [dict(name="synthetic_nominal", task_type="classification")], tuned=False)
    group("revision_stress", "alignment_synthetic_reliability.yaml", load_config(REPO / "configs/t1/alignment_synthetic_reliability.yaml")["datasets"], tuned=False)
    # Radius fallbacks run only after primary train-calibration diagnostics
    # establish the need; real participants are never replaced by simulation.
    if args.groups:
        groups = [g for g in groups if g[0] in args.groups]
    write_json(out / "execution_plan.json", dict(source_sha256=source_fingerprint(REPO), grid_study=str(grid / "study.json"),
        jobs=[dict(group=name, config=str(config), dataset=ds, seed=seed) for name, config, ds, seed in groups],
        automated_only=True, human_participant_study="requires actual approved participants",
        general_maintenance_formula="not selected by this automation"))
    if args.plan_only:
        print(f"Prepared {len(groups)} dataset/seed jobs", flush=True)
        return
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(execute, config, ds, seed, out): name for name, config, ds, seed in groups}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            write_json(out / "progress.json", dict(expected_jobs=len(groups), finished_jobs=len(results),
                passed=sum(r["exit_code"] == 0 for r in results), failed=[r for r in results if r["exit_code"] != 0]))
            print(f"[{len(results)}/{len(groups)}] {future_map[future]} {result['dataset']['name']} s{result['seed']}: exit {result['exit_code']}", flush=True)
    write_json(out / "execution_results.json", results)
    if any(r["exit_code"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
