"""Run the maintained NN-kNN classification benchmarks as run units under ``runs/core/<study>/``.

Every (dataset, method, split) is one unit ``runs/core/<study>/<dataset>/<method>/<mode>-<index>``
(model/common/tracking.py): finals ``accuracy`` (and ``best_metric`` for NN-kNN), the
``predictions`` array, and for NN-kNN the per-epoch curves and the named states of
``train_model``. Rerunning the same ``--study`` skips complete units and resumes incomplete
NN-kNN units from their last checkpoint. The summary is printed and recorded in the report unit
``runs/core/<study>/_reports/summary/<timestamp>``.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from model.classification_workflow import (  # noqa: E402
    make_classification_cfg,
    run_repeated_classification_model_benchmarks,
    summarize_classification_benchmark_runs,
)
from model.common.tracking import RunKey, load_run, load_state, open_run, unit_dir  # noqa: E402
from tools.table1_nnknn_kfold import plain_config, record_frame, unit_component, unit_scope, utc_stamp  # noqa: E402

IMAGE_NAMES = {"mnist", "cifar10", "cifar-10", "cifar_10", "svhn"}


def benchmark_suite(datasets: list[str]) -> str:
    names = {name.lower() for name in datasets}
    return "image" if names <= IMAGE_NAMES else "mixed" if names & IMAGE_NAMES else "tabular"


def run_benchmarks(
    datasets: list[str],
    cfg: Mapping[str, Any],
    *,
    study: str,
    methods: list[str] | None = None,
    method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "holdout",
    runs: int = 5,
    seed: int = 42,
    dataset_kwargs_map: Mapping[str, Mapping[str, Any]] | None = None,
    root: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run (or resume) a study; returns the summary and per-run tables, both read from the event
    files of the units, and the unit log (``run_id``, ``path``, ``source``) in sweep order.
    ``method_cfgs`` are the per-method baseline settings of the workflow (for example
    ``{"convnet": {"epochs": 5}}``); they are part of every unit's config."""
    unit_log: list[dict[str, Any]] = []
    keys: list[tuple[RunKey, str]] = []

    def unit_key(state: Mapping[str, Any], method: str) -> RunKey:
        return RunKey("core", study, (unit_component(state["display_name"]), unit_component(method),
                                      f"{state['run_mode']}-{int(state['run_index'])}"))

    def unit_state(state: Mapping[str, Any], method: str, name: str) -> Any:
        """A named state of a complete unit (the shared image classifier), or None."""
        data = load_run(unit_dir(unit_key(state, method), root), arrays=False)
        return load_state(data.path, name) if name in data.states() else None

    @contextlib.contextmanager
    def unit_run(state: Mapping[str, Any], method: str) -> Iterator[Any]:
        name, index, run_mode = str(state["requested_name"]), int(state["run_index"]), str(state["run_mode"])
        key = unit_key(state, method)
        config = {
            "kind": "classification_benchmark", "dataset": str(state["display_name"]), "method": method,
            "mode": run_mode, "requested_mode": mode, "run_index": index, "run_seed": int(state["run_seed"]),
            # a k-fold split depends on the fold count and the split seed; a holdout or official split only on
            # run_seed, so a study can later be extended with more runs
            **({"num_runs": runs, "base_seed": seed} if run_mode == "kfold" else {}),
            "nnknn_cfg": plain_config(cfg) if method.startswith("nnknn") else None,
            "dataset_kwargs": plain_config(dict((dataset_kwargs_map or {}).get(name, {}) or {})),
            **({"method_cfgs": plain_config(dict(method_cfgs))} if method_cfgs else {}),
        }
        keys.append((key, run_mode))
        with unit_scope(key, config, root=root, log=unit_log) as run:
            yield run
        print(f"[classification benchmark] {key.run_id}: {unit_log[-1]['source']}", flush=True)

    run_repeated_classification_model_benchmarks(
        datasets,
        cfg,
        methods=methods,
        method_cfgs=method_cfgs,
        num_runs=runs,
        mode=mode,
        base_seed=seed,
        dataset_kwargs_map=dataset_kwargs_map,
        unit_run=unit_run,
        unit_state=unit_state,
    )
    rows = []
    for key, run_mode in keys:  # the per-run table, read back from the event files
        data = load_run(unit_dir(key, root), arrays=False)
        rows.append({"dataset": data.config["dataset"], "mode": run_mode, "method": data.config["method"],
                     "run_index": data.config["run_index"], "run_seed": data.config["run_seed"],
                     "accuracy": data.final()["accuracy"]})
    runs_df = pd.DataFrame(rows)
    return summarize_classification_benchmark_runs(runs_df), runs_df, unit_log


def record_summary_report(
    summary: pd.DataFrame,
    runs_df: pd.DataFrame,
    *,
    study: str,
    manifest: Mapping[str, Any],
    units: list[dict[str, Any]],
    root: Any = None,
) -> Path:
    """Record the tables in ``runs/core/<study>/_reports/summary/<timestamp>`` (UTC): row records
    ``summary`` and ``runs`` (column order in ``columns/summary`` and ``columns/runs``; read them back
    with ``tools.table1_nnknn_kfold.read_frame``), the record ``manifest`` and ``units`` (run id and
    result digest)."""
    bound = [{"run_id": u["run_id"], "result_digest": load_run(u["path"], arrays=False).info.result_digest}
             for u in units]
    key = RunKey("core", study, ("_reports", "summary", utc_stamp()))
    manifest = plain_config(dict(manifest))
    with open_run(key, config={"kind": "classification_report", "manifest": manifest, "units": bound}, root=root) as run:
        record_frame(run, "summary", summary)
        record_frame(run, "runs", runs_df)
        run.log_record("manifest", manifest)
        run.log_record("units", bound)
    return unit_dir(key, root)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run maintained NN-kNN classification benchmarks.")
    parser.add_argument("datasets", nargs="+", help="Dataset names, such as iris digits cifar10.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to benchmark.")
    parser.add_argument("--mode", default="holdout", choices=["holdout", "kfold", "official", "preset"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalizer", default="softmax", choices=["softmax", "sparsemax"])
    parser.add_argument("--study", default=None,
                        help="Study name under runs/core/. Defaults to classification_<suite>_<timestamp>; "
                             "pass an existing study to skip its complete units and resume the others.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = make_classification_cfg(
        {
            "training_epochs": args.epochs,
            "batch_size": args.batch_size,
            "case_normalizer": args.normalizer,
        }
    )
    dataset_kwargs_map = {
        name: {
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
        }
        for name in args.datasets
        if name.lower() in IMAGE_NAMES
    }
    suite = benchmark_suite(args.datasets)
    created_at = datetime.now(timezone.utc)
    study = args.study or f"classification_{suite}_{created_at.strftime('%Y%m%d_%H%M%S')}"
    print(f"Study: runs/core/{study}", flush=True)
    summary, runs_df, unit_log = run_benchmarks(
        args.datasets,
        cfg,
        study=study,
        methods=args.methods,
        mode=args.mode,
        runs=args.runs,
        seed=args.seed,
        dataset_kwargs_map=dataset_kwargs_map,
    )
    manifest = {
        "suite": suite,
        "datasets": args.datasets,
        "methods": args.methods,
        "mode": args.mode,
        "runs": args.runs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "normalizer": args.normalizer,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    report = record_summary_report(summary, runs_df, study=study, manifest=manifest, units=unit_log)
    print(summary.to_string(index=False))
    print(f"Recorded the summary in {report}.")
    return report


if __name__ == "__main__":
    main()
