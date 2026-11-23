"""Run the maintained NN-kNN classification benchmarks.

The summary goes to a fresh folder ``<--output-dir>/classification_<suite>_<timestamp>/`` with
``summary.csv``, ``runs.csv`` and ``manifest.json``. With ``--study S`` the training data of every
(dataset, method, split) is also recorded as a run unit
``NN-KNN/runs/core/S/<dataset>/<method>/<mode>-<index>`` (model/common/tracking.py): finals
``accuracy`` (and ``best_metric`` for NN-kNN), the ``predictions`` array, and for NN-kNN the
per-epoch curves and the named states of ``train_model`` (no checkpoint files). Rerunning the same
study skips complete units, taking their recorded accuracy for the summary rows, and resumes
incomplete NN-kNN units from their last checkpoint. The summary files are the same either way.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from model.classification_workflow import (  # noqa: E402
    make_classification_cfg,
    run_repeated_classification_model_benchmarks,
)
from model.common.tracking import RunKey, load_run, open_run, resolve_unit, unit_dir  # noqa: E402


def unit_component(name: Any) -> str:
    """A dataset or method name as a run-unit path component."""
    return re.sub(r"[^A-Za-z0-9._=+-]+", "_", str(name)).strip("._") or "unnamed"


def plain_config(obj: Any) -> Any:
    """JSON view of a config for ``rec/config`` (tensors as lists, devices as str, no ``checkpoint_path``)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (torch.device, Path)):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): plain_config(v) for k, v in obj.items() if k != "checkpoint_path"}
    if isinstance(obj, (list, tuple)):
        return [plain_config(v) for v in obj]
    return obj


def study_unit_run(
    study: str,
    cfg: Mapping[str, Any],
    *,
    mode: str,
    runs: int,
    seed: int,
    method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_kwargs_map: Mapping[str, Mapping[str, Any]] | None = None,
    root: Any = None,
    log: list[dict[str, Any]] | None = None,
):
    """The ``unit_run`` callback of ``run_repeated_classification_model_benchmarks`` that records
    every (dataset, method, split) under ``runs/core/<study>/``. It yields the open run of an
    incomplete unit (completed when the method returns, marked failed if it raises) or the
    ``RunData`` of a complete one. ``log`` collects ``{run_id, path, source}`` per unit, ``source``
    being ``complete``, ``fresh``, ``resumed`` or ``restarted``. A complete unit written with
    another config raises ``ConfigMismatchError``."""

    @contextlib.contextmanager
    def unit_run(state: Mapping[str, Any], method: str) -> Iterator[Any]:
        name, index, run_mode = str(state["requested_name"]), int(state["run_index"]), str(state["run_mode"])
        key = RunKey("core", study, (unit_component(state["display_name"]), unit_component(method),
                                     f"{run_mode}-{index}"))
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
        entry = {"run_id": key.run_id, "path": unit_dir(key, root), "source": "complete"}
        if log is not None:
            log.append(entry)
        if resolve_unit(key, config=config, root=root) is not None:
            yield load_run(entry["path"], arrays=False)
        else:
            with open_run(key, config=config, root=root) as run:
                entry["source"] = "restarted" if run.restarted else ("resumed" if run.resumed_from else "fresh")
                yield run
        print(f"[classification benchmark] {key.run_id}: {entry['source']}", flush=True)

    return unit_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run maintained NN-kNN classification benchmarks.")
    parser.add_argument("datasets", nargs="+", help="Dataset names, such as iris digits cifar10.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to benchmark.")
    parser.add_argument("--mode", default="holdout", choices=["holdout", "kfold", "official", "preset"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalizer", default="softmax", choices=["softmax", "sparsemax"])
    parser.add_argument("--output-dir", default="results", help="Parent folder for a fresh timestamped run.")
    parser.add_argument("--study", default=None,
                        help="Also record the training data of every dataset, method and split as run units under "
                             "NN-KNN/runs/core/<study>/; rerunning a study skips its complete units and resumes "
                             "the others. Off by default.")
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
        if name.lower() in {"mnist", "cifar10", "cifar-10", "svhn"}
    }
    summary, runs, _ = run_repeated_classification_model_benchmarks(
        args.datasets,
        cfg,
        methods=args.methods,
        num_runs=args.runs,
        mode=args.mode,
        base_seed=args.seed,
        dataset_kwargs_map=dataset_kwargs_map,
        unit_run=None if args.study is None else study_unit_run(
            args.study, cfg, mode=args.mode, runs=args.runs, seed=args.seed,
            dataset_kwargs_map=dataset_kwargs_map),
    )
    image_names = {"mnist", "cifar10", "cifar-10", "cifar_10", "svhn"}
    normalized_names = {name.lower() for name in args.datasets}
    suite = "image" if normalized_names <= image_names else "mixed" if normalized_names & image_names else "tabular"
    created_at = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir) / f"classification_{suite}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    runs.to_csv(output_dir / "runs.csv", index=False)
    manifest = {
        "created_at_utc": created_at.isoformat(),
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
        "outputs": ["summary.csv", "runs.csv", "manifest.json"],
    }
    if args.study is not None:
        manifest["study"] = args.study
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote results to {output_dir}.")


if __name__ == "__main__":
    main()
