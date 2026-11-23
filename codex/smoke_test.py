from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_import_smoke() -> None:
    from model.regression_workflow import (
        list_supported_regression_benchmark_methods,
        list_supported_regression_datasets,
        make_regression_cfg,
    )
    from model.classification_workflow import (
        list_supported_classification_benchmark_methods,
        list_supported_classification_datasets,
    )

    datasets = list_supported_regression_datasets()
    methods = list_supported_regression_benchmark_methods()
    cls_datasets = list_supported_classification_datasets()
    cls_methods = list_supported_classification_benchmark_methods()
    regression_cfg = make_regression_cfg({"task_type": "regression"})
    if regression_cfg.get("case_score_mode") != "bias_minus_distance":
        raise AssertionError("Generic regression runs must use the current bias-minus-distance case score.")
    if regression_cfg.get("normalize_over_cases") is not True:
        raise AssertionError("Generic regression runs must normalize case activations.")
    print("import smoke ok")
    print(f"synthetic datasets: {datasets['synthetic']}")
    print(f"benchmark methods: {methods}")
    print(f"classification small datasets: {cls_datasets['small']}")
    print(f"classification methods: {cls_methods}")


def _smoke_unit(study: str, name: str, cfg: dict):
    """The run unit of one smoke mode, ``runs/core/<study>/smoke/<name>``, and its RunData if it is complete
    (recording is mandatory; a complete unit is never trained again)."""
    from model.common.tracking import RunKey, load_unit
    from tools.run_classification_benchmarks import plain_config

    key = RunKey("core", study, ("smoke", name))
    config = {"kind": "smoke_test", "mode": name, "cfg": plain_config(cfg)}
    return key, config, load_unit(key, config=config, arrays=False)


def run_training_smoke(study: str) -> None:
    from model.common.tracking import open_run
    from model.regression_workflow import make_regression_cfg, run_single_nnknn_regression_experiment

    cfg = make_regression_cfg(
        {
            "task_type": "regression",
            "training_epochs": 1,
            "batch_size": 32,
            "top_k": 8,
            "bias_manual_set": True,
            "bias_manual_value": 1.0,
        }
    )
    key, config, done = _smoke_unit(study, "train", cfg)
    if done is not None:
        final = done.final()
        print(f"training smoke: {key.run_id} is complete, not retrained")
        print(f"rmse_raw={final['rmse_raw']:.6f}")
        print(f"best_metric={final['best_metric']:.6f}")
        return
    with open_run(key, config=config) as run:
        state = run_single_nnknn_regression_experiment(
            "linear_regression",
            cfg,
            dataset_kwargs={"n": 96, "d": 5, "noise_scale": 0.1},
            run_seed=0,
            dataset_seed=0,
            split_seed=0,
            standardize=False,
            run=run,
        )
        rmse_raw = float(state["rmse_raw"])
        best_metric = float(state["best_metric"])
        run.log_final({"rmse_raw": rmse_raw, "best_metric": best_metric})
    print("training smoke ok")
    print(f"rmse_raw={rmse_raw:.6f}")
    print(f"best_metric={best_metric:.6f}")


def run_classification_smoke(study: str) -> None:
    from model.classification_workflow import make_classification_cfg, run_single_nnknn_classification_experiment
    from model.common.tracking import open_run

    cfg = make_classification_cfg(
        {
            "training_epochs": 1,
            "batch_size": 32,
            "top_k": 3,
            "explanation_mode": True,
            "case_normalizer": "softmax",
        }
    )
    key, config, done = _smoke_unit(study, "classification", cfg)
    if done is not None:
        print(f"classification smoke: {key.run_id} is complete, not retrained")
        print(f"accuracy={done.final()['accuracy']:.6f}")
        return
    with open_run(key, config=config) as run:
        state = run_single_nnknn_classification_experiment(
            "iris",
            cfg,
            run_seed=0,
            split_seed=0,
            run=run,
        )
        probabilities = state["class_probabilities"]
        if not torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.size(0)), atol=1e-5):
            raise AssertionError("Classification outputs do not sum to one.")
        if state.get("most_activated_cases") is None:
            raise AssertionError("Classification explanation outputs were not returned.")
        if state.get("most_activated_class_ids") is None:
            raise AssertionError("Classification explanation labels were not decoded.")
        run.log_final({"accuracy": float(state["accuracy"])})
    print("classification training smoke ok")
    print(f"accuracy={float(state['accuracy']):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for Codex cloud environments.")
    parser.add_argument(
        "--mode",
        default="imports",
        choices=("imports", "train", "classification"),
        help="Choose a lightweight import check or a tiny training run.",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Required for train and classification: the training is recorded as the run unit "
        "NN-KNN/runs/core/<study>/smoke/<mode> (a complete unit is reported, not retrained).",
    )
    args = parser.parse_args()
    if args.mode != "imports" and args.study is None:
        parser.error(f"--mode {args.mode} requires --study (recording is mandatory)")

    if args.mode == "train":
        run_training_smoke(args.study)
    elif args.mode == "classification":
        run_classification_smoke(args.study)
    else:
        run_import_smoke()


if __name__ == "__main__":
    main()
