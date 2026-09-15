from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.t1_maintenance import get_maintenance_policy
from model.t1_mcb import compute_neighborhood_churn, compute_representation_drift
from model.t1_synchronization import calibrate_free_correction_radius
from model.t1_workflow import (
    T1Config,
    build_t1_model,
    evaluate_t1_model,
    run_t1_maintenance_step,
    train_leave_one_out_adapter,
    write_t1_experiment_artifacts,
)


def load_dataset(name: str, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Loads and standardizes a benchmark dataset."""
    name = name.lower()
    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == "synthetic":
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=3,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    num_classes = len(np.unique(y))
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=num_classes).float()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=num_classes).float()

    return X_train_t, y_train_t, X_test_t, y_test_t, num_classes


def run_single_t1_experiment(
    dataset_name: str,
    policy_name: str,
    adapter_mode: str,
    mcb_enabled: bool,
    case_capacity: int = 30,
    seed: int = 42,
    output_root: str | Path = "results/t1_benchmark",
) -> dict[str, Any]:
    """Runs a single matched-condition T1 experiment."""
    X_train, y_train, X_test, y_test, num_classes = load_dataset(dataset_name, seed=seed)

    use_adapter = adapter_mode != "none"
    cfg = T1Config(
        case_maintenance_policy=policy_name,
        case_capacity=case_capacity,
        mcb_enabled=mcb_enabled,
        classification_adapter_enabled=use_adapter,
        classification_adapter_output_mode=adapter_mode if use_adapter else "nominal_residual_scores",
        seed=seed,
    )

    # 1. Build T1 model
    model, stats_store, archive_store, adapter = build_t1_model(X_train, y_train, cfg)

    # 2. Simulate training-time retrieval exposure
    model.eval()
    with torch.no_grad():
        res = model(X_train)
        probs = res[0]
        preds = probs.argmax(dim=-1)
        targets = y_train.argmax(dim=-1)

        for i in range(X_train.size(0)):
            is_corr = bool((preds[i] == targets[i]).item())
            stats_store.record_retrieval(
                case_id=i,
                activation=float(probs[i, preds[i]].item()),
                is_correct=is_corr,
                step=10,
            )

    # 3. Perform maintenance
    policy = get_maintenance_policy(policy_name, seed=seed)
    removed, actions = run_t1_maintenance_step(
        model=model,
        stats_store=stats_store,
        archive_store=archive_store,
        policy=policy,
        target_capacity=case_capacity,
        step=50,
    )

    # 4. Calibrate free-correction radius
    calib = calibrate_free_correction_radius(
        model=model,
        cases=model.cases[:model.case_count()],
        labels=model.labels[:model.case_count()],
        max_pairs=500,
        seed=seed,
    )

    # 5. Train adapter if enabled
    if adapter is not None:
        train_leave_one_out_adapter(
            model=model,
            adapter=adapter,
            train_queries=X_train,
            train_labels=y_train,
            epochs=30,
            lr=5e-3,
        )

    # 6. Evaluate
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    metrics, retrieval_events = evaluate_t1_model(
        model=model,
        test_loader=test_loader,
        adapter=adapter,
        log_retrievals=True,
    )

    # 7. Write contract artifacts
    run_dir = Path(output_root) / f"{dataset_name}_{policy_name}_{adapter_mode}_mcb{int(mcb_enabled)}_s{seed}"
    write_t1_experiment_artifacts(
        output_dir=run_dir,
        cfg=cfg,
        eval_metrics=metrics,
        stats_store=stats_store,
        archive_store=archive_store,
        maintenance_log=actions,
        retrieval_events=retrieval_events,
        calibration_result=calib,
    )

    result_summary = {
        "dataset": dataset_name,
        "policy": policy_name,
        "adapter": adapter_mode,
        "mcb": mcb_enabled,
        "seed": seed,
        "active_cases": model.case_count(),
        "archived_cases": archive_store.count(),
        "pre_accuracy": metrics["pre_accuracy"],
        "post_accuracy": metrics["post_accuracy"],
        "accuracy_gain": metrics["accuracy_gain"],
        "total_flips": metrics["total_flips"],
        "correct_flips": metrics["correct_flips"],
        "harmful_flips": metrics["harmful_flips"],
        "net_flip_benefit": metrics["net_flip_benefit"],
        "tau_task": calib.tau_task,
        "run_dir": str(run_dir),
    }
    return result_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1 Neural CBR Benchmark Matrix")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostic subset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--output-dir", type=str, default="results/t1_benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["iris", "synthetic"] if args.quick else ["iris", "wine", "synthetic"]
    policies = [
        "full_memory",
        "random",
        "stratified",
        "bias_only",
        "provenance_only",
        "trustworthiness_only",
        "provenance_bias_coverage",
    ]
    adapters = ["none", "nominal_residual_scores", "logit_residual"]
    mcb_modes = [False, True]

    print(f"=== Starting T1 Full-Cycle CBR Benchmark Suite ===")
    print(f"Datasets: {datasets}")
    print(f"Policies: {policies}")
    print(f"Adapters: {adapters}")
    print(f"MCB Conditions: {mcb_modes}")
    print(f"Seeds: {args.seeds}\n")

    results: list[dict[str, Any]] = []
    total_runs = len(datasets) * len(policies) * len(adapters) * len(mcb_modes) * len(args.seeds)
    run_idx = 0

    for ds in datasets:
        for pol in policies:
            for ad in adapters:
                for mcb in mcb_modes:
                    for s in args.seeds:
                        run_idx += 1
                        print(f"[{run_idx}/{total_runs}] ds={ds} pol={pol} ad={ad} mcb={mcb} seed={s} ...", end=" ", flush=True)
                        t0 = time.time()
                        try:
                            res = run_single_t1_experiment(
                                dataset_name=ds,
                                policy_name=pol,
                                adapter_mode=ad,
                                mcb_enabled=mcb,
                                case_capacity=20 if ds == "iris" else 40,
                                seed=s,
                                output_root=out_dir,
                            )
                            dt = time.time() - t0
                            print(f"DONE ({dt:.1f}s) | Pre-Acc: {res['pre_accuracy']:.3f} Post-Acc: {res['post_accuracy']:.3f} Flips: +{res['correct_flips']}/-{res['harmful_flips']}")
                            results.append(res)
                        except Exception as e:
                            print(f"FAILED: {e}")

    # Write summary CSV
    summary_csv = out_dir / "benchmark_summary.csv"
    if results:
        fieldnames = list(results[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"\n=== Benchmark Complete! Saved {len(results)} runs to {summary_csv} ===")


if __name__ == "__main__":
    main()
