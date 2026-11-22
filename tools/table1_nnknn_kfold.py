"""Table 1 k-fold benchmark (NN-kNN families and baselines).

With ``study`` set, every (dataset, family or method, fold) is one run unit under
``runs/core/<study>/<dataset>/<family|method>/f<fold>`` (model/common/tracking.py): a complete
unit is skipped and its rows are read back from its event file, an incomplete NN-kNN unit
resumes training from its last checkpoint, and the aggregate tables go to a report unit
``runs/core/<study>/_reports/table1/<timestamp>``. Without ``study`` nothing is recorded.
"""
from __future__ import annotations

import contextlib
import copy
from datetime import datetime, timezone
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from model.common.tracking import NullRun, RunKey, load_run, load_state, open_run, resolve_unit, unit_dir
from model.nn_cdh import NNCDHAdapter
from model.nnknn_model import GlocalFeatureWeight, NN_KNN_Model, device
from model.regression_workflow import (
    configure_regression_cfg_for_state,
    evaluate_nnknn_pre_post_adaptation_state,
    load_regression_dataset_state,
    make_regression_cfg,
    run_repeated_regression_model_benchmarks,
    split_regression_state,
    standardize_regression_state,
    summarize_regression_benchmark_runs,
    train_nnknn_regression_state,
)

TABLE1_DEFAULT_DATASET_NAMES: list[str] = [
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


def build_table1_paper_base_cfg(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared NN-kNN Table 1 config described in the paper.

    Notes:
    - This is for the tabular Table 1 setting, not the UTKFace setup.
    - Parameters explicitly stated in the paper are set here.
    - Parameters not stated in the paper keep the repository defaults.
    """
    cfg = make_regression_cfg(
        {
            "task_type": "regression",
            "case_score_mode": "bias_minus_distance",
            "normalize_over_cases": True,
            "tau": 1.0,
            "regression_locality": False,
            "lambda_base": 1.0,
            "locality_alpha": 2.0,
            "lambda_expdist": 0.1,
            "eps_sigma_multiplier": 0.1,
            "training_epochs": 7000,
            "batch_size": 64,
            "feature_extractor_lr": 1e-3,
            "glocal_weightor_lr": 1e-3,
            "case_net_lr": 3e-4,
            "checkpoint_path": "nnknn_regression_best.pth",
            "patience": 80,
            "post_mlp_enabled": False,
            "explanation_mode": True,
        }
    )
    if overrides:
        cfg.update(copy.deepcopy(dict(overrides)))
    return cfg


def make_table1_family_cfg(
    *,
    case_normalizer: str,
    use_locality: bool,
    base_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one of the four Table 1 NN-kNN configuration families.

    Each family is trained with NN-CDH enabled so the resulting workflow state
    can report both:
    - pre-adaptation RMSE (`pure` / retrieval-only)
    - post-adaptation RMSE (`adaptation`)

    This matches the paper's family structure:
    softmax, softmax+locality, sparsemax, sparsemax+locality.
    """
    cfg = build_table1_paper_base_cfg(base_cfg)
    cfg.update(
        {
            "case_normalizer": case_normalizer,
            "regression_locality": bool(use_locality),
            "use_nn_cdh": True,
            "nn_cdh_pretrain": True,
            "cdh_aggregate": True,
        }
    )
    return cfg


TABLE1_NNKNN_FAMILIES: list[dict[str, Any]] = [
    {
        "family_id": "softmax",
        "family_label": "NN-kNN (softmax)",
        "case_normalizer": "softmax",
        "use_locality": False,
    },
    {
        "family_id": "softmax_locality",
        "family_label": "NN-kNN (softmax) + locality",
        "case_normalizer": "softmax",
        "use_locality": True,
    },
    {
        "family_id": "sparsemax",
        "family_label": "NN-kNN (sparsemax)",
        "case_normalizer": "sparsemax",
        "use_locality": False,
    },
    {
        "family_id": "sparsemax_locality",
        "family_label": "NN-kNN (sparsemax) + locality",
        "case_normalizer": "sparsemax",
        "use_locality": True,
    },
]

TABLE1_BASELINE_METHODS: list[str] = ["knn_x", "mlkr_knn", "mlp"]

TABLE1_TRANSPOSED_DATASET_LABELS: dict[str, str] = {
    "abalone": "Abalone",
    "airfoil": "Airfoil",
    "bike_sharing": "Bike Sharing",
    "body_fat": "Body Fat",
    "califonia_housing": "California Housing",
    "car": "Cars",
    "diabets": "Diabetes",
    "energy_efficiency": "Energy Efficiency",
    "student_performance": "Student Performance",
    "wine": "Wine Quality",
    "yacht": "Yacht",
}

TABLE1_TRANSPOSED_MODEL_LABELS: dict[str, str] = {
    "kNN(X)": "Weighted k-NN",
    "MLP": "MLP",
    "MLKR+kNN": "MLKR",
    "NN-kNN (softmax) - pure": "NN-kNN (softmax), pure",
    "NN-kNN (softmax) - adaptation": "NN-kNN (softmax), adaptation",
    "NN-kNN (softmax) + locality - locality": "NN-kNN (softmax) + locality, locality",
    "NN-kNN (softmax) + locality - locality + adaptation": "NN-kNN (softmax) + locality, locality + adaptation",
    "NN-kNN (sparsemax) - pure": "NN-kNN (sparsemax), pure",
    "NN-kNN (sparsemax) - adaptation": "NN-kNN (sparsemax), adaptation",
    "NN-kNN (sparsemax) + locality - locality": "NN-kNN (sparsemax) + locality, locality",
    "NN-kNN (sparsemax) + locality - locality + adaptation": "NN-kNN (sparsemax) + locality, locality + adaptation",
}

TABLE1_TRANSPOSED_MODEL_ORDER: list[str] = [
    "Weighted k-NN",
    "MLP",
    "MLKR",
    "NN-kNN (softmax), pure",
    "NN-kNN (softmax), adaptation",
    "NN-kNN (softmax) + locality, locality",
    "NN-kNN (softmax) + locality, locality + adaptation",
    "NN-kNN (sparsemax), pure",
    "NN-kNN (sparsemax), adaptation",
    "NN-kNN (sparsemax) + locality, locality",
    "NN-kNN (sparsemax) + locality, locality + adaptation",
]


def _clone_state_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in bundle.items()
    }


def _paper_result_name(*, use_locality: bool, adapted: bool) -> str:
    if use_locality:
        return "locality + adaptation" if adapted else "locality"
    return "adaptation" if adapted else "pure"


def build_table1_baseline_method_cfgs(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the baseline configs used by the notebook's repeated benchmark cell."""
    cfgs: dict[str, dict[str, Any]] = {
        "knn_x": {"k": 5, "batch_size": 512, "device": torch.device("cpu")},
        "mlkr_knn": {"k": 25, "n_components": None, "max_iter": 1000},
        "mlp": {
            "hidden": (64, 32),
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "val_batch_size": 512,
            "epochs": 200,
            "patience": 20,
            "grad_clip_norm": 1.0,
            "device": torch.device("cpu"),
        },
    }
    for method_name, method_overrides in (overrides or {}).items():
        cfgs.setdefault(method_name, {})
        cfgs[method_name].update(copy.deepcopy(dict(method_overrides)))
    return cfgs


def _normalize_table1_nnknn_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    normalized = summary_df.copy()
    normalized["source"] = "nnknn"
    normalized["entry_id"] = normalized["family_id"] + ":" + normalized["result_type"]
    normalized["entry_label"] = normalized["family_label"] + " - " + normalized["result_label"]
    return normalized


def _normalize_table1_nnknn_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    normalized = runs_df.copy()
    normalized["source"] = "nnknn"
    normalized["entry_id"] = normalized["family_id"] + ":" + normalized["result_type"]
    normalized["entry_label"] = normalized["family_label"] + " - " + normalized["result_label"]
    return normalized


def _normalize_table1_baseline_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in summary_df.to_dict("records"):
        method = row["method"]
        method_label = row.get("method_label") or method
        records.append(
            {
                "dataset": row["dataset"],
                "family_id": method,
                "family_label": method_label,
                "result_type": "baseline",
                "result_label": method_label,
                "num_folds": row["num_runs"],
                "rmse_raw_mean": row.get("rmse_raw_mean"),
                "rmse_raw_std": row.get("rmse_raw_std"),
                "rmse_raw_table": row.get("rmse_raw_table"),
                "mode": row.get("mode"),
                "source": "baseline",
                "entry_id": method,
                "entry_label": method_label,
            }
        )
    return pd.DataFrame(records)


def _normalize_table1_baseline_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in runs_df.to_dict("records"):
        method = row["method"]
        method_label = row.get("method_label") or method
        records.append(
            {
                "dataset": row["dataset"],
                "mode": row["mode"],
                "fold": row["fold"],
                "run_index": row["run_index"],
                "family_id": method,
                "family_label": method_label,
                "result_type": "baseline",
                "result_label": method_label,
                "rmse_raw": row.get("rmse_raw"),
                "rmse_model_space": row.get("rmse_model_space"),
                "standardized_targets": row.get("standardized_targets"),
                "source": "baseline",
                "entry_id": method,
                "entry_label": method_label,
            }
        )
    return pd.DataFrame(records)


def summarize_table1_nnknn_runs(
    runs_df: pd.DataFrame,
    *,
    digits: int = 4,
) -> pd.DataFrame:
    """Aggregate fold-level Table 1 NN-kNN results into paper-style summaries."""
    if runs_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    grouped = runs_df.groupby(
        ["dataset", "family_id", "family_label", "result_type", "result_label"],
        dropna=False,
        sort=False,
    )

    for (dataset, family_id, family_label, result_type, result_label), group in grouped:
        rmse_values = group["rmse_raw"].dropna()
        if rmse_values.empty:
            continue

        rmse_mean = float(rmse_values.mean())
        rmse_std = float(rmse_values.std(ddof=1)) if len(rmse_values) > 1 else 0.0
        records.append(
            {
                "dataset": dataset,
                "family_id": family_id,
                "family_label": family_label,
                "result_type": result_type,
                "result_label": result_label,
                "num_folds": int(len(group)),
                "rmse_raw_mean": rmse_mean,
                "rmse_raw_std": rmse_std,
                "rmse_raw_table": f"{rmse_mean:.{digits}f} +/- {rmse_std:.{digits}f}",
            }
        )

    return pd.DataFrame(records)


def _make_table1_split_state(
    dataset_state: Mapping[str, Any],
    *,
    train_idx: Any,
    val_idx: Any,
    fold_idx: int,
    base_seed: int,
) -> dict[str, Any]:
    split_state = split_regression_state(
        _clone_state_bundle(dataset_state),
        seed=base_seed,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    split_state = standardize_regression_state(split_state, enabled=True)
    split_state.update(
        {
            "dataset": split_state["display_name"],
            "mode": "kfold",
            "run_seed": base_seed + fold_idx - 1,
            "dataset_seed": base_seed,
            "split_seed": base_seed,
            "run_index": fold_idx,
            "fold": fold_idx,
        }
    )
    return split_state


def unit_component(name: Any) -> str:
    """A dataset, family or method name as a run-unit path component."""
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


@contextlib.contextmanager
def unit_scope(key: RunKey, config: Mapping[str, Any], *, root: Any = None,
               log: list[dict[str, Any]] | None = None) -> Iterator[Any]:
    """Yield None for a complete unit (skip it), else an open run that completes when the block ends
    (and records the failure if the block raises). ``log`` collects ``{run_id, path, source}``."""
    info = resolve_unit(key, config=config, root=root)
    entry = {"run_id": key.run_id, "path": unit_dir(key, root), "source": "complete"}
    if log is not None:
        log.append(entry)
    if info is not None:
        yield None
        return
    with open_run(key, config=config, root=root) as run:
        entry["source"] = "restarted" if run.restarted else ("resumed" if run.resumed_from else "fresh")
        yield run


def _seed_unit(seed: int) -> None:
    """Each tracked unit starts from its own seed, so skipping or resuming a unit leaves the others unchanged."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_table1_nnknn_run_state(
    split_state: Mapping[str, Any],
    cfg_run: Mapping[str, Any],
    run_path: str | Path,
    *,
    feature_extractor: Any = None,
) -> dict[str, Any]:
    """Rebuild a Table 1 NN-kNN workflow state from the ``final`` state of a complete unit."""
    run_feature_extractor = copy.deepcopy(feature_extractor) if feature_extractor is not None else None
    feature_dim = (
        split_state["X_train"].shape[-1]
        if run_feature_extractor is None
        else run_feature_extractor.feature_dim
    )
    glocal_weightor = GlocalFeatureWeight(feature_dim, cfg_run["glocal_fw_set_num"]).to(device)
    adapter = NNCDHAdapter(feature_dim, label_dim=1).to(device) if cfg_run.get("use_nn_cdh", False) else None
    labels = split_state["y_train_norm"].float().unsqueeze(1)

    model = NN_KNN_Model(
        split_state["X_train"],
        labels,
        feature_extractor=run_feature_extractor,
        glocal_weightor=glocal_weightor,
        nn_cdh=adapter,
        **dict(cfg_run),
    ).to(device)
    model.load_state_dict(load_state(run_path, "final", map_location=device))

    workflow_state = dict(split_state)
    workflow_state.update(
        {
            "cfg": copy.deepcopy(dict(cfg_run)),
            "cfg_run": copy.deepcopy(dict(cfg_run)),
            "feature_extractor": feature_extractor,
            "glocal_weightor": glocal_weightor,
            "model": model,
            "run_path": str(run_path),
        }
    )
    return workflow_state


# column order of the NN-kNN fold rows (row records come back with their keys sorted)
_NNKNN_RUN_COLUMNS = ["dataset", "mode", "fold", "run_index", "family_id", "family_label", "standardized_targets",
                      "resume_source", "result_type", "result_label", "rmse_raw", "rmse_model_space"]


def _ordered_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """``df`` with ``columns`` first, in that order (those present), then any others."""
    first = [c for c in columns if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _append_table1_nnknn_run_records(
    runs: list[dict[str, Any]],
    *,
    dataset_name: str,
    fold_idx: int,
    family: Mapping[str, Any],
    workflow_state: Mapping[str, Any],
    resume_source: str | None = None,
) -> None:
    base_record = {
        "dataset": dataset_name,
        "mode": "kfold",
        "fold": fold_idx,
        "run_index": fold_idx,
        "family_id": family["family_id"],
        "family_label": family["family_label"],
        "standardized_targets": workflow_state["standardized_targets"],
    }
    if resume_source is not None:
        base_record["resume_source"] = resume_source

    for adapted in (False, True):
        result_name = _paper_result_name(
            use_locality=bool(family["use_locality"]),
            adapted=adapted,
        )
        runs.append(
            {
                **base_record,
                "result_type": result_name,
                "result_label": result_name,
                "rmse_raw": workflow_state["rmse_post_raw"] if adapted else workflow_state["rmse_pre_raw"],
                "rmse_model_space": (
                    workflow_state["rmse_post_model_space"]
                    if adapted
                    else workflow_state["rmse_pre_model_space"]
                ),
            }
        )


def _train_table1_nnknn_unit(
    run: Any,
    split_state: Mapping[str, Any],
    cfg_run: Mapping[str, Any],
    *,
    dataset_name: str,
    fold_idx: int,
    family: Mapping[str, Any],
    feature_extractor: Any,
) -> dict[str, Any]:
    """Train and evaluate one family on one fold; with a real run, record its rows, finals and predictions."""
    workflow_state = train_nnknn_regression_state(
        split_state,
        cfg_run,
        feature_extractor=feature_extractor,
        run=run,
        resumable=True,
    )
    workflow_state = evaluate_nnknn_pre_post_adaptation_state(
        workflow_state,
        batch_size=512,
        show_plots=False,
        print_metrics=False,
    )
    rows: list[dict[str, Any]] = []
    _append_table1_nnknn_run_records(
        rows, dataset_name=dataset_name, fold_idx=fold_idx, family=family, workflow_state=workflow_state
    )
    if not isinstance(run, NullRun):
        run.log_records("rows", rows)
        run.log_final({key: workflow_state[key] for key in (
            "rmse_pre_raw", "rmse_post_raw", "rmse_pre_model_space", "rmse_post_model_space", "best_metric")})
        for key in ("y_pred_pre_z", "y_pred_post_z"):
            run.log_array(key, workflow_state[key].numpy())  # CPU tensors made by the evaluation
    return workflow_state


def _table1_nnknn_state_from_unit(
    split_state: Mapping[str, Any],
    cfg_run: Mapping[str, Any],
    data: Any,
    *,
    feature_extractor: Any,
    rebuild: bool,
) -> dict[str, Any]:
    """Workflow state of a complete (skipped) unit: its finals and predictions from the event file; the
    model is rebuilt from its ``final`` state only when ``rebuild`` is set."""
    if rebuild:
        workflow_state = load_table1_nnknn_run_state(split_state, cfg_run, data.path, feature_extractor=feature_extractor)
    else:
        workflow_state = dict(split_state)
        workflow_state["run_path"] = str(data.path)
    workflow_state.update(data.final())
    for key in ("y_pred_pre_z", "y_pred_post_z"):
        workflow_state[key] = torch.from_numpy(data.array(key))
    return workflow_state


def run_table1_nnknn_kfold(
    dataset_names: str | list[str],
    *,
    num_folds: int = 5,
    base_seed: int = 42,
    base_cfg: Mapping[str, Any] | None = None,
    feature_extractor: Any = None,
    study: str | None = None,
    root: Any = None,
    unit_log: list[dict[str, Any]] | None = None,
    rebuild_skipped: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Run the Table 1 NN-kNN families with k-fold CV.

    Output structure:
    - `summary_df`: one row per dataset/family/result_type
    - `runs_df`: one row per dataset/family/fold/result_type
    - `artifacts[dataset][family_id]`: trained workflow states per fold

    With `study`, each (dataset, family, fold) is the unit
    `runs/core/<study>/<dataset>/<family_id>/f<fold>`: it starts from the seed
    `base_seed + fold - 1`, a complete unit is not retrained, an incomplete one resumes, and
    `runs_df` is read from the event files. The artifact of a complete unit holds its finals
    (`rmse_*`, `best_metric`), `y_pred_pre_z` / `y_pred_post_z` and `run_path`, read from the
    event file; its model is rebuilt from the `final` state only with `rebuild_skipped=True`.
    `unit_log` collects `{run_id, path, source}` per unit (source: complete, fresh, resumed,
    restarted).
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    runs: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    total_datasets = len(dataset_names)
    unit_log = [] if unit_log is None else unit_log

    for dataset_idx, dataset_name in enumerate(dataset_names, start=1):
        print(
            f"[table1][nnknn] Dataset {dataset_idx}/{total_datasets}: {dataset_name}",
            flush=True,
        )
        dataset_state = load_regression_dataset_state(dataset_name)
        artifacts[dataset_state["display_name"]] = {
            family["family_id"]: [] for family in TABLE1_NNKNN_FAMILIES
        }

        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=base_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dataset_state["X"]), start=1):
            print(
                f"[table1][nnknn] {dataset_state['display_name']} fold {fold_idx}/{num_folds}",
                flush=True,
            )
            split_state = _make_table1_split_state(
                dataset_state,
                train_idx=train_idx,
                val_idx=val_idx,
                fold_idx=fold_idx,
                base_seed=base_seed,
            )

            for family in TABLE1_NNKNN_FAMILIES:
                print(
                    f"[table1][nnknn] {dataset_state['display_name']} fold {fold_idx}/{num_folds} family={family['family_id']}",
                    flush=True,
                )
                family_cfg = make_table1_family_cfg(
                    case_normalizer=family["case_normalizer"],
                    use_locality=family["use_locality"],
                    base_cfg=base_cfg,
                )
                cfg_run = configure_regression_cfg_for_state(family_cfg, split_state)
                unit = dict(dataset_name=dataset_state["display_name"], fold_idx=fold_idx, family=family)
                if study is None:
                    workflow_state = _train_table1_nnknn_unit(
                        NullRun(), split_state, cfg_run, feature_extractor=feature_extractor, **unit)
                    _append_table1_nnknn_run_records(runs, workflow_state=workflow_state, **unit)
                else:
                    key = RunKey("core", study, (
                        unit_component(dataset_state["display_name"]), unit_component(family["family_id"]), f"f{fold_idx}"))
                    config = {
                        "kind": "table1_nnknn", "dataset": dataset_state["display_name"], "fold": fold_idx,
                        "num_folds": num_folds, "base_seed": base_seed, "family": plain_config(family),
                        "cfg": plain_config(cfg_run), "seed": split_state["run_seed"],
                        "feature_extractor": None if feature_extractor is None else type(feature_extractor).__name__,
                    }
                    workflow_state = None
                    with unit_scope(key, config, root=root, log=unit_log) as run:
                        if run is not None:
                            _seed_unit(split_state["run_seed"])
                            workflow_state = _train_table1_nnknn_unit(
                                run, split_state, cfg_run, feature_extractor=feature_extractor, **unit)
                    print(f"[table1][nnknn] {key.run_id}: {unit_log[-1]['source']}", flush=True)
                    data = load_run(unit_dir(key, root), arrays=workflow_state is None)
                    if workflow_state is None:  # complete unit: skipped
                        workflow_state = _table1_nnknn_state_from_unit(
                            split_state, cfg_run, data, feature_extractor=feature_extractor, rebuild=rebuild_skipped)
                    runs.extend(data.records("rows"))
                workflow_state.update(
                    {
                        "family_id": family["family_id"],
                        "family_label": family["family_label"],
                    }
                )
                artifacts[dataset_state["display_name"]][family["family_id"]].append(workflow_state)

    runs_df = _ordered_columns(pd.DataFrame(runs), _NNKNN_RUN_COLUMNS)
    summary_df = summarize_table1_nnknn_runs(runs_df)
    return summary_df, runs_df, artifacts


def _resume_info(unit_log: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {source: sum(e["source"] == source for e in unit_log) for source in ("complete", "fresh", "resumed", "restarted")}
    return {"units": [{"run_id": e["run_id"], "source": e["source"]} for e in unit_log], **counts}


def run_table1_nnknn_kfold_resumable(
    dataset_names: str | list[str],
    *,
    study: str,
    num_folds: int = 5,
    base_seed: int = 42,
    base_cfg: Mapping[str, Any] | None = None,
    feature_extractor: Any = None,
    root: Any = None,
    rebuild_skipped: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """`run_table1_nnknn_kfold` on an existing study: complete units are skipped, incomplete ones
    resume from their last checkpoint. The last item counts the units by source."""
    unit_log: list[dict[str, Any]] = []
    summary_df, runs_df, artifacts = run_table1_nnknn_kfold(
        dataset_names, num_folds=num_folds, base_seed=base_seed, base_cfg=base_cfg,
        feature_extractor=feature_extractor, study=study, root=root, unit_log=unit_log,
        rebuild_skipped=rebuild_skipped)
    return summary_df, runs_df, artifacts, _resume_info(unit_log)


def _table1_baselines(
    dataset_names: str | list[str],
    *,
    num_folds: int,
    base_seed: int,
    baseline_method_cfgs: Mapping[str, Mapping[str, Any]] | None,
    feature_extractor: Any,
    study: str | None,
    root: Any,
    unit_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    method_cfgs = build_table1_baseline_method_cfgs(baseline_method_cfgs)
    unit_run: Callable[[Mapping[str, Any], str], Any] | None = None
    keys: list[RunKey] = []
    if study is not None:
        @contextlib.contextmanager
        def unit_run(state: Mapping[str, Any], method: str) -> Iterator[Any]:
            key = RunKey("core", study, (unit_component(state["display_name"]), unit_component(method), f"f{state['fold']}"))
            config = {"kind": "table1_baseline", "dataset": state["display_name"], "method": method,
                      "fold": state["fold"], "num_folds": num_folds, "base_seed": base_seed, "mode": "kfold",
                      "standardize": "auto", "method_cfg": plain_config(method_cfgs.get(method, {})),
                      "feature_extractor": None if feature_extractor is None else type(feature_extractor).__name__}
            keys.append(key)
            with unit_scope(key, config, root=root, log=unit_log) as run:
                yield run
            print(f"[table1][baseline] {key.run_id}: {unit_log[-1]['source']}", flush=True)

    summary_df, runs_df, artifacts = run_repeated_regression_model_benchmarks(
        dataset_names=dataset_names,
        nnknn_cfg=None,
        feature_extractor=feature_extractor,
        methods=TABLE1_BASELINE_METHODS,
        method_cfgs=method_cfgs,
        num_runs=num_folds,
        mode="kfold",
        base_seed=base_seed,
        standardize="auto",
        unit_run=unit_run,
    )
    if study is not None:  # every row from its unit's event file, in sweep order
        runs_df = pd.DataFrame([load_run(unit_dir(key, root), arrays=False).record("row") for key in keys])
        summary_df = summarize_regression_benchmark_runs(runs_df) if not runs_df.empty else pd.DataFrame()
    return summary_df, runs_df, artifacts


def _combine_table1(
    baseline_summary_df: pd.DataFrame,
    baseline_runs_df: pd.DataFrame,
    nnknn_summary_df: pd.DataFrame,
    nnknn_runs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.concat(
        [
            _normalize_table1_baseline_summary(baseline_summary_df),
            _normalize_table1_nnknn_summary(nnknn_summary_df),
        ],
        ignore_index=True,
        sort=False,
    )
    runs_df = pd.concat(
        [
            _normalize_table1_baseline_runs(baseline_runs_df),
            _normalize_table1_nnknn_runs(nnknn_runs_df),
        ],
        ignore_index=True,
        sort=False,
    )
    return summary_df, runs_df


def run_table1_kfold(
    dataset_names: str | list[str],
    *,
    num_folds: int = 5,
    base_seed: int = 42,
    nnknn_base_cfg: Mapping[str, Any] | None = None,
    baseline_method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    feature_extractor: Any = None,
    study: str | None = None,
    root: Any = None,
    run_baselines: bool = True,
    unit_log: list[dict[str, Any]] | None = None,
    rebuild_skipped: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the full Table 1 suite for k-fold CV, excluding UTKFace if omitted upstream.

    Returns:
    - `summary_df`: normalized combined summary for NN-kNN families and baselines.
    - `runs_df`: normalized fold-level results for NN-kNN families and baselines.
    - `artifacts`: nested dict containing raw NN-kNN and baseline artifacts.

    With `study`, every NN-kNN family and baseline method per dataset and fold is a run unit
    (see the module docstring); complete units are skipped, incomplete ones resume
    (`rebuild_skipped` as in `run_table1_nnknn_kfold`).
    """
    unit_log = [] if unit_log is None else unit_log
    print("[table1] Starting NN-kNN family sweep", flush=True)
    nnknn_summary_df, nnknn_runs_df, nnknn_artifacts = run_table1_nnknn_kfold(
        dataset_names,
        num_folds=num_folds,
        base_seed=base_seed,
        base_cfg=nnknn_base_cfg,
        feature_extractor=feature_extractor,
        study=study,
        root=root,
        unit_log=unit_log,
        rebuild_skipped=rebuild_skipped,
    )

    baseline_summary_df, baseline_runs_df, baseline_artifacts = pd.DataFrame(), pd.DataFrame(), {}
    if run_baselines:
        print("[table1] Starting baseline sweep", flush=True)
        baseline_summary_df, baseline_runs_df, baseline_artifacts = _table1_baselines(
            dataset_names, num_folds=num_folds, base_seed=base_seed, baseline_method_cfgs=baseline_method_cfgs,
            feature_extractor=feature_extractor, study=study, root=root, unit_log=unit_log)
    else:
        print("[table1] Skipping baseline sweep", flush=True)

    summary_df, runs_df = _combine_table1(baseline_summary_df, baseline_runs_df, nnknn_summary_df, nnknn_runs_df)
    print("[table1] Finished full Table 1 sweep", flush=True)
    return summary_df, runs_df, {"baselines": baseline_artifacts, "nnknn": nnknn_artifacts}


def run_table1_kfold_resumable(
    dataset_names: str | list[str],
    *,
    study: str,
    num_folds: int = 5,
    base_seed: int = 42,
    nnknn_base_cfg: Mapping[str, Any] | None = None,
    baseline_method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    feature_extractor: Any = None,
    run_baselines: bool = True,
    root: Any = None,
    rebuild_skipped: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """`run_table1_kfold` on an existing study: complete units (NN-kNN and baselines) are skipped
    and read back, incomplete NN-kNN units resume from their last checkpoint."""
    unit_log: list[dict[str, Any]] = []
    summary_df, runs_df, artifacts = run_table1_kfold(
        dataset_names, num_folds=num_folds, base_seed=base_seed, nnknn_base_cfg=nnknn_base_cfg,
        baseline_method_cfgs=baseline_method_cfgs, feature_extractor=feature_extractor, study=study,
        root=root, run_baselines=run_baselines, unit_log=unit_log, rebuild_skipped=rebuild_skipped)
    resume_info = {"base_seed": base_seed, "num_folds": num_folds, "run_baselines": run_baselines,
                   **_resume_info(unit_log)}
    return summary_df, runs_df, artifacts, resume_info


def build_table1_like_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the wide dataset-by-entry Table 1 layout."""
    if summary_df.empty:
        return pd.DataFrame(columns=["dataset"])
    return summary_df.pivot(index="dataset", columns="entry_label", values="rmse_raw_table").reset_index()


def build_table1_transposed_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the transposed Table 1 layout used for spreadsheet-friendly comparison."""
    if summary_df.empty:
        return pd.DataFrame(columns=["Model"])

    transposed = summary_df.loc[:, ["dataset", "entry_label", "rmse_raw_table"]].copy()
    transposed["dataset_label"] = transposed["dataset"].map(TABLE1_TRANSPOSED_DATASET_LABELS).fillna(transposed["dataset"])
    transposed["model_label"] = transposed["entry_label"].map(TABLE1_TRANSPOSED_MODEL_LABELS).fillna(transposed["entry_label"])

    transposed_df = transposed.pivot(
        index="model_label",
        columns="dataset_label",
        values="rmse_raw_table",
    )

    ordered_models = [label for label in TABLE1_TRANSPOSED_MODEL_ORDER if label in transposed_df.index]
    ordered_models.extend(label for label in transposed_df.index if label not in ordered_models)

    ordered_dataset_labels = [
        TABLE1_TRANSPOSED_DATASET_LABELS[dataset_name]
        for dataset_name in TABLE1_DEFAULT_DATASET_NAMES
        if TABLE1_TRANSPOSED_DATASET_LABELS[dataset_name] in transposed_df.columns
    ]
    ordered_dataset_labels = sorted(ordered_dataset_labels)
    ordered_dataset_labels.extend(label for label in transposed_df.columns if label not in ordered_dataset_labels)

    transposed_df = transposed_df.reindex(index=ordered_models, columns=ordered_dataset_labels)
    transposed_df.index.name = "Model"
    return transposed_df.reset_index()


_REPORT_TABLES = ("summary_long", "runs_long", "table1_like", "transposed")


def _table_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): (None if v is None or v is pd.NA else v) for k, v in row.items()} for row in df.to_dict("records")]


def record_frame(run: Any, name: str, df: pd.DataFrame) -> None:
    """Record ``df`` as the row record ``name`` and its column order as the record ``columns/<name>``
    (row records come back with their keys sorted); ``read_frame`` restores both."""
    run.log_records(name, _table_rows(df))
    run.log_record(f"columns/{name}", [str(c) for c in df.columns])


def read_frame(data: Any, name: str) -> pd.DataFrame:
    """A table recorded by ``record_frame``, with its rows and its column order."""
    return pd.DataFrame(data.records(name), columns=data.record(f"columns/{name}"))


def utc_stamp(fmt: str = "%Y%m%d-%H%M%S-%f") -> str:
    """Timestamp (UTC) used in study and report unit names."""
    return datetime.now(timezone.utc).strftime(fmt)


def record_table1_report(
    summary_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    *,
    study: str,
    metadata: Mapping[str, Any] | None = None,
    units: list[dict[str, Any]] | None = None,
    root: Any = None,
) -> Path:
    """Record the Table 1 tables in the report unit `runs/core/<study>/_reports/table1/<timestamp>`.

    Row records `summary_long`, `runs_long`, `table1_like` and `transposed` (one row per table
    row), `meta` (`metadata`, row counts) and `units` (run id and result digest of every unit the
    tables were read from), and each table's column order (`columns/<table>`). Returns the unit
    directory; `read_table1_report` reads it back.
    """
    tables = {
        "summary_long": summary_df,
        "runs_long": runs_df,
        "table1_like": build_table1_like_df(summary_df),
        "transposed": build_table1_transposed_df(summary_df),
    }
    bound = [{"run_id": u["run_id"], "result_digest": load_run(u["path"], arrays=False).info.result_digest}
             for u in units or []]
    meta = {"summary_rows": int(len(summary_df)), "run_rows": int(len(runs_df)), **plain_config(dict(metadata or {}))}
    key = RunKey("core", study, ("_reports", "table1", utc_stamp()))
    with open_run(key, config={"kind": "table1_report", "meta": meta, "units": bound}, root=root) as run:
        for name, df in tables.items():
            record_frame(run, name, df)
        run.log_record("meta", meta)
        run.log_record("units", bound)
    return unit_dir(key, root)


def read_table1_report(path: str | Path) -> dict[str, pd.DataFrame]:
    """The tables of a complete report unit written by `record_table1_report`, columns in their order."""
    data = load_run(path, arrays=False)
    return {name: read_frame(data, name) for name in _REPORT_TABLES}
