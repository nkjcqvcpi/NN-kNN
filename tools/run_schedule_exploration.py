from __future__ import annotations

import argparse
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
from model.t1_scheduler import FullPipelineTrainer, TrainingScheduleResult, TrainingScheduleType
from model.t1_workflow import T1Config, build_t1_model


def load_dataset(name: str, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
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


def run_schedule_comparison(
    dataset_name: str,
    schedules: list[str],
    case_capacity: int = 30,
    total_epochs: int = 40,
    mcb_enabled: bool = True,
    seeds: list[int] = [42, 43],
    output_dir: str | Path = "results/t1_schedules",
) -> list[dict[str, Any]]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict[str, Any]] = []

    for seed in seeds:
        X_train, y_train, X_test, y_test, num_classes = load_dataset(dataset_name, seed=seed)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        for sched in schedules:
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

            cfg = T1Config(
                case_maintenance_policy="provenance_bias_coverage",
                case_capacity=case_capacity,
                mcb_enabled=mcb_enabled,
                mcb_momentum=0.95,
                classification_adapter_enabled=True,
                classification_adapter_output_mode="nominal_residual_scores",
                seed=seed,
            )

            # Build fresh pipeline instance
            model, stats_store, archive_store, adapter = build_t1_model(X_train, y_train, cfg)
            policy = get_maintenance_policy(cfg.case_maintenance_policy, seed=seed)

            trainer = FullPipelineTrainer(
                model=model,
                adapter=adapter,
                stats_store=stats_store,
                archive_store=archive_store,
                policy=policy,
                cfg=cfg,
            )

            res = trainer.train_schedule(
                train_loader=train_loader,
                val_loader=val_loader,
                schedule=sched,
                total_epochs=total_epochs,
                maintenance_checkpoint_epoch=total_epochs // 3,
            )

            rec = {
                "dataset": dataset_name,
                "schedule": sched,
                "seed": seed,
                "mcb": mcb_enabled,
                "total_epochs": total_epochs,
                "best_pre_acc": res.best_pre_acc,
                "best_post_acc": res.best_post_acc,
                "final_pre_acc": res.final_pre_acc,
                "final_post_acc": res.final_post_acc,
                "accuracy_gain": res.accuracy_gain,
                "final_net_flips": res.final_net_flips,
                "tau_task": res.calibration_tau,
                "final_rep_drift": res.final_rep_drift,
                "time_sec": res.total_time_sec,
            }
            summary_records.append(rec)

            # Save detailed history
            run_file = out_path / f"{dataset_name}_{sched}_s{seed}.json"
            with run_file.open("w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)

    return summary_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Optimal Training Schedules for T1 Neural CBR")
    parser.add_argument("--datasets", nargs="+", default=["iris", "wine", "synthetic"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--output-dir", type=str, default="results/t1_schedules")
    args = parser.parse_args()

    schedules = [
        TrainingScheduleType.STAGED_SEQUENTIAL.value,
        TrainingScheduleType.ALTERNATING.value,
        TrainingScheduleType.JOINT_SYNCHRONIZED.value,
        TrainingScheduleType.WARMUP_ALTERNATING.value,
        TrainingScheduleType.WARMUP_JOINT.value,
    ]

    print("=================================================================")
    print("      T1 Neural CBR Full-Pipeline Training Schedule Exploration   ")
    print("=================================================================")
    print(f"Datasets: {args.datasets}")
    print(f"Candidate Schedules: {schedules}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Output Directory: {args.output_dir}\n")

    all_records: list[dict[str, Any]] = []

    for ds in args.datasets:
        cap = 20 if ds == "iris" else 40
        print(f"\n>>> Running Benchmark Family on Dataset: {ds.upper()} (Capacity K={cap}) <<<")
        records = run_schedule_comparison(
            dataset_name=ds,
            schedules=schedules,
            case_capacity=cap,
            total_epochs=args.epochs,
            mcb_enabled=True,
            seeds=args.seeds,
            output_dir=args.output_dir,
        )
        all_records.extend(records)

    # Save summary CSV
    out_csv = Path(args.output_dir) / "schedule_comparison_summary.csv"
    if all_records:
        fieldnames = list(all_records[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)

    print(f"\n=== Exploration Complete! Saved {len(all_records)} schedule records to {out_csv} ===")

    # Aggregate & print comparison table
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in all_records:
        k = (r["dataset"], r["schedule"])
        grouped.setdefault(k, []).append(r)

    print("\n" + "=" * 95)
    print(f"{'Dataset':12s} | {'Schedule':22s} | {'Pre-Acc (mean)':15s} | {'Post-Acc (mean)':16s} | {'Gain':8s} | {'Net Flips':10s}")
    print("=" * 95)
    for (ds, sched), items in sorted(grouped.items()):
        pre_m = float(np.mean([float(x["final_pre_acc"]) for x in items]))
        post_m = float(np.mean([float(x["final_post_acc"]) for x in items]))
        gain_m = float(np.mean([float(x["accuracy_gain"]) for x in items]))
        flips_m = float(np.mean([int(x["final_net_flips"]) for x in items]))
        print(f"{ds:12s} | {sched:22s} | {pre_m:.3f}           | {post_m:.3f}            | {gain_m:+.3f}   | {flips_m:+4.1f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
