"""Run the Table 1 5-fold regression benchmark as run units under ``runs/core/<study>/``.

Every (dataset, NN-kNN family or baseline method, fold) is one unit; rerunning the same
``--study`` skips complete units and resumes incomplete ones. The tables are printed and
recorded in the report unit ``runs/core/<study>/_reports/table1/<timestamp>``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.table1_nnknn_kfold import (  # noqa: E402
    build_table1_transposed_df,
    record_table1_report,
    run_table1_kfold,
    utc_stamp,
)


DEFAULT_TABLE1_DATASETS = [
    "califonia_housing",
    "diabets",
    "abalone",
    "body_fat",
    "airfoil",
    "car",
    "student_performance",
    "yacht",
    "energy_efficiency",
    "bike_sharing",
    "wine",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Table 1 5-fold regression benchmark suite."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_TABLE1_DATASETS,
        help="Dataset keys to run. Defaults to the current Table 1 subset.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds. Default: 5.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Default: 42.",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Study name under runs/core/. Defaults to table1_kfold_<timestamp>; pass an existing "
             "study to skip its complete units and resume the others.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    study = args.study or f"table1_kfold_{utc_stamp('%Y%m%d_%H%M%S')}"  # UTC

    print("Starting Table 1 k-fold sweep", flush=True)
    print(f"Study: runs/core/{study}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"num_folds={args.num_folds} base_seed={args.base_seed}", flush=True)

    unit_log: list[dict] = []
    summary_df, runs_df, _ = run_table1_kfold(
        dataset_names=args.datasets,
        num_folds=args.num_folds,
        base_seed=args.base_seed,
        study=study,
        unit_log=unit_log,
    )
    report = record_table1_report(
        summary_df,
        runs_df,
        study=study,
        metadata={"datasets": list(args.datasets), "num_folds": int(args.num_folds), "base_seed": int(args.base_seed)},
        units=unit_log,
    )

    print("Finished Table 1 k-fold sweep", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(build_table1_transposed_df(summary_df).to_string(index=False), flush=True)
    print(f"Report unit: {report}", flush=True)
    return report


if __name__ == "__main__":
    main()
