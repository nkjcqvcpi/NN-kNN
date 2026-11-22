"""Resume a Table 1 k-fold study under ``runs/core/<study>/``.

Complete units are skipped and read back from their event files, incomplete NN-kNN units
resume from their last checkpoint, the others run from scratch. The tables are printed and
recorded in a new report unit ``runs/core/<study>/_reports/table1/<timestamp>``; which units were
skipped, resumed or run is printed (and recorded per unit in its ``_run/resume`` records).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.table1_nnknn_kfold import (  # noqa: E402
    TABLE1_DEFAULT_DATASET_NAMES,
    build_table1_transposed_df,
    record_table1_report,
    run_table1_kfold,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a Table 1 NN-kNN k-fold study: skip complete units, resume the others.",
    )
    parser.add_argument(
        "--study",
        required=True,
        help="Existing study name under runs/core/ (for example table1_kfold_20260421_114800).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used for the k-fold split. Must match the study (units check their config).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds. Must match the study (units check their config).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to run. Defaults to the repository Table 1 dataset list.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only run/resume NN-kNN entries; do not run the Table 1 baselines.",
    )
    parser.add_argument(
        "--scikit-learn-data",
        type=Path,
        default=None,
        help="Writable sklearn dataset cache directory. Defaults to scikit-learn's own location ($SCIKIT_LEARN_DATA if set, else ~/scikit_learn_data).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)

    # Library downloads stay in the library's default location: only an explicit
    # --scikit-learn-data overrides it; otherwise scikit-learn picks its own cache
    # ($SCIKIT_LEARN_DATA if set, else ~/scikit_learn_data).
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.scikit_learn_data is not None:
        args.scikit_learn_data.mkdir(parents=True, exist_ok=True)
        os.environ["SCIKIT_LEARN_DATA"] = str(args.scikit_learn_data)

    dataset_names = args.datasets or TABLE1_DEFAULT_DATASET_NAMES

    print("Resuming Table 1 study", flush=True)
    print("Study: runs/core/" + args.study, flush=True)
    print("Datasets:", dataset_names, flush=True)
    print("Base seed:", args.base_seed, flush=True)

    unit_log: list[dict] = []
    summary_df, runs_df, _ = run_table1_kfold(
        dataset_names=dataset_names,
        num_folds=args.num_folds,
        base_seed=args.base_seed,
        study=args.study,
        run_baselines=not args.skip_baselines,
        unit_log=unit_log,
    )
    counts = {source: sum(u["source"] == source for u in unit_log) for source in ("complete", "resumed", "restarted", "fresh")}
    report = record_table1_report(
        summary_df,
        runs_df,
        study=args.study,
        metadata={"datasets": list(dataset_names), "num_folds": int(args.num_folds), "base_seed": int(args.base_seed),
                  "run_baselines": not args.skip_baselines, "resumed_invocation": True, "unit_sources": counts},
        units=unit_log,
    )
    print(summary_df.to_string(index=False), flush=True)
    print(build_table1_transposed_df(summary_df).to_string(index=False), flush=True)
    print("Units:", ", ".join(f"{n} {source}" for source, n in counts.items()), flush=True)
    print(f"Report unit: {report}", flush=True)
    print("Finished Table 1 resume", flush=True)
    return report


if __name__ == "__main__":
    main()
