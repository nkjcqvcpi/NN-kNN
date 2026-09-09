from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from model.t0_maintenance import write_maintenance_artifacts
from model.t0_workflow import (
    T0Config,
    build_t0_model,
    evaluate_t0_model,
    run_t0_maintenance_step,
    train_leave_one_out_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T0 Full-Cycle Neural CBR Benchmark Driver.")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "iris", "wine"])
    parser.add_argument("--policy", default="provenance_bias_coverage",
                        choices=["provenance_bias_coverage", "bias_only", "provenance_only", "trustworthiness_only", "stratified"])
    parser.add_argument("--case-capacity", type=int, default=150)
    parser.add_argument("--target-capacity", type=int, default=50)
    parser.add_argument("--adapter", action="store_true", help="Enable classification NN-CDH reuse adapter.")
    parser.add_argument("--adapter-mode", default="nominal_residual_scores", choices=["nominal_residual_scores", "logit_residual"])
    parser.add_argument("--adapter-epochs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5, help="Trustworthiness weight alpha.")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Provenance quality smoothing s.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/t0_benchmark")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_data(dataset_name: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=600,
            n_features=16,
            n_informative=10,
            n_redundant=4,
            n_classes=3,
            random_state=seed,
        )
    elif dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif dataset_name == "wine":
        data = load_wine()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    num_classes = int(len(np.unique(y)))

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    return X_tr_t, y_tr_t, X_val_t, y_val_t, num_classes


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # 1. Load dataset
    X_tr, y_tr, X_val, y_val, num_classes = load_data(args.dataset, args.seed)
    
    # Cap initial case base to case_capacity if needed
    if X_tr.size(0) > args.case_capacity:
        X_tr = X_tr[:args.case_capacity]
        y_tr = y_tr[:args.case_capacity]

    # 2. Configure T0
    cfg = T0Config(
        task_type="classification",
        case_capacity=X_tr.size(0),
        target_capacity=args.target_capacity,
        case_maintenance_policy=args.policy,
        case_score_smoothing=args.smoothing,
        case_trust_alpha=args.alpha,
        classification_adapter_enabled=args.adapter,
        classification_adapter_output_mode=args.adapter_mode,
        adapter_epochs=args.adapter_epochs,
        seed=args.seed,
        device=args.device,
    )

    print(f"[T0] Dataset: {args.dataset} | Initial cases: {X_tr.size(0)} | Target capacity: {args.target_capacity}")
    print(f"[T0] Retention policy: {args.policy} | Adapter enabled: {args.adapter} ({args.adapter_mode})")

    # 3. Build T0 model
    model, stats_store, archive_store, policy = build_t0_model(X_tr, y_tr, cfg)
    model.to(device)

    # 4. Observe training retrievals to populate provenance counters
    _ = evaluate_t0_model(model, X_tr, y_tr, cfg, stats_store=stats_store, step=0, device=device)

    # 5. Baseline pre-maintenance validation evaluation
    pre_maint_eval = evaluate_t0_model(model, X_val, y_val, cfg, device=device)
    print(f"[T0] Pre-maintenance validation accuracy: {pre_maint_eval['pre_accuracy']:.4f}")

    # 6. Execute maintenance step (retention policy selection down to target_capacity)
    new_count, actions = run_t0_maintenance_step(
        model, stats_store, archive_store, policy, target_capacity=args.target_capacity, step=1
    )
    print(f"[T0] Executed maintenance: active cases reduced from {X_tr.size(0)} to {new_count}")

    # 7. If adapter enabled, train adapter on leave-one-out neighborhoods of retained cases
    if args.adapter:
        active_X = model.cases[:new_count]
        active_y = model.labels[:new_count]
        train_result = train_leave_one_out_adapter(model, active_X, active_y, cfg, device=device)
        print(f"[T0] Trained classification adapter ({args.adapter_mode}): final_loss={train_result.get('final_loss', 0.0):.4f}")

    # 8. Post-maintenance evaluation
    post_maint_eval = evaluate_t0_model(model, X_val, y_val, cfg, device=device)
    print(f"[T0] Post-maintenance retrieval accuracy: {post_maint_eval['pre_accuracy']:.4f}")
    if args.adapter:
        print(f"[T0] Post-maintenance adapted accuracy:   {post_maint_eval['post_accuracy']:.4f}")
        print(f"[T0] Decision flips: total={post_maint_eval['total_flips']}, correct={post_maint_eval['correct_flips']}, harmful={post_maint_eval['harmful_flips']}, net benefit={post_maint_eval['net_flip_benefit']}")

    # 9. Write artifacts
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.policy}_{run_timestamp}"
    write_maintenance_artifacts(out_dir, stats_store, actions, archive_store)

    summary = {
        "dataset": args.dataset,
        "policy": args.policy,
        "target_capacity": args.target_capacity,
        "adapter_enabled": args.adapter,
        "adapter_mode": args.adapter_mode,
        "pre_maintenance_accuracy": pre_maint_eval["pre_accuracy"],
        "post_maintenance_retrieval_accuracy": post_maint_eval["pre_accuracy"],
        "post_maintenance_adapted_accuracy": post_maint_eval["post_accuracy"] if args.adapter else None,
        "flip_diagnostics": post_maint_eval if args.adapter else None,
        "archived_cases": archive_store.count(),
        "retained_cases": new_count,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[T0] Results and maintenance logs saved to: {out_dir}")


if __name__ == "__main__":
    main()
