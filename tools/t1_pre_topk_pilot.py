"""Small paired mask ablation, retaining an identical nearest-neighbor bias initializer."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["NNKNN_DEVICE"] = "cpu"

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from model.common.artifacts import environment_info, git_state, source_fingerprint, write_json
from model.common.core import CoreConfig, build_model, evaluate, train_retrieval
from model.regression_workflow import get_regression_dataset, prepare_regression_split_for_training
from model.t1.data import T1Data, make_splits


def data_for(ds, seed, splits):
    if ds["task_type"] == "classification":
        return make_splits(ds["name"], "classification", seed, **splits)
    frame = None
    if ds["name"] == "student_performance":
        raw = pd.read_csv(REPO / "datasets/student-por.csv", sep=";")
        frame, y = raw.drop(columns="G3"), torch.tensor(raw.G3.to_numpy(), dtype=torch.float32)
    else:
        bundle = get_regression_dataset(ds["name"], **({**ds["generator"], "seed": seed} if "generator" in ds else {}))
        X, y = bundle["X"], bundle["y"].view(-1)
    tr, te = train_test_split(np.arange(len(y)), test_size=splits["test_frac"], random_state=seed)
    tr, va = train_test_split(tr, test_size=splits["val_frac"] / (1 - splits["test_frac"]), random_state=seed + 1)
    if frame is not None:
        categorical = frame.select_dtypes(exclude="number").columns.tolist()
        encoder = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                      categorical)], remainder="passthrough", sparse_threshold=0.0)
        encoder.fit(frame.iloc[tr])
        X = torch.tensor(encoder.transform(frame).astype(np.float32))
    ready = prepare_regression_split_for_training(X[tr], y[tr], X[va], y[va], standardize=True)
    # Apply the same training-only statistics to the untouched test stream.
    Xte = (X[te] - ready["X_mean"]) / ready["X_std"]
    yte = ((y[te].view(-1, 1) - ready["y_mean"]) / ready["y_std"]).view(-1)
    return T1Data(ds["name"], "regression", ready["X_train"], ready["y_train"], ready["X_val"],
                  ready["y_val"], Xte, yte, y_scale=(float(ready["y_mean"].item()), float(ready["y_std"].item())),
                  meta={"source": "model.regression_workflow", "split_indices": {"train": tr.tolist(),
                        "validation": va.tolist(), "test": te.tolist()}, "preprocessing_fit_stream": "train_only",
                        **({"categorical_encoding": "training-fitted one-hot, then training-only standardization"}
                           if frame is not None else {})})


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def support(model, data, training):
    model.train(training)
    X = data.X_train[:16] if training else data.X_test[:16]
    ids = torch.arange(len(X)) if training else None
    w = model.retrieve(X, exclude_identical=False, query_case_ids=ids)["weights"]
    if training:
        assert torch.count_nonzero(w[torch.arange(len(X)), ids]) == 0, "LOO exclusion failed"
    return {"positive_weights_mean": float((w > 0).sum(1).float().mean()),
            "positive_weights_max": int((w > 0).sum(1).max()),
            "effective_neighbors_mean": float((1.0 / w.square().sum(1)).mean())}


def summarize(study, rows, out, config_path):
    summary = []
    for ds in study["datasets"]:
        r = [x for x in rows if x["dataset"] == ds["name"]]
        masked = [next(x for x in r if x["seed"] == s and x["masked"]) for s in study["seeds"]]
        full = [next(x for x in r if x["seed"] == s and not x["masked"]) for s in study["seeds"]]
        a, b = np.array([x["test_error"] for x in masked]), np.array([x["test_error"] for x in full])
        summary.append({"dataset": ds["name"], "task_type": ds["task_type"], "n_pairs": len(a),
                        "masked_mean_error": float(a.mean()), "masked_sd_error": float(a.std(ddof=1)),
                        "all_case_mean_error": float(b.mean()), "all_case_sd_error": float(b.std(ddof=1)),
                        "relative_error_change_percent": float(100 * (b.mean() / a.mean() - 1)) if a.mean() else None,
                        "mean_paired_change": float((b - a).mean()), "unmasked_better_pairs": int((b < a).sum()),
                        "unmasked_worse_pairs": int((b > a).sum()), "ties": int((b == a).sum())})
    write_json(out / "summary.json", summary)
    dataset_count, seed_count = len(study["datasets"]), len(study["seeds"])
    core = study["core"]
    unmasked = [r for r in rows if not r["masked"]]
    cap_count = sum(r["best_epoch"] == core["epochs"] for r in unmasked)
    better = sum(s["all_case_mean_error"] < s["masked_mean_error"] for s in summary)
    lines = ["# Pre-Top-K mask comparison, October 30, 2026", "",
             f"Exploratory retrieval-only comparison: {dataset_count} datasets, {seed_count} paired seeds "
             f"{study['seeds']}, {len(rows)} fits. "
             "No tuning, reuse adapter, retention, or confirmatory claim.", "",
             "Only `pre_topk_mask` changes between arms. `top_k=5` sets truncation in the masked arm; "
             "`bias_init_k=5` independently sets the nearest-neighbor bias initializer in both arms. "
             "Each pair starts with exactly identical model state and bias (SHA-256 checked).", "",
             "60% train / 20% validation / 20% test; preprocessing uses train only; training uses identity "
             f"leave-one-out; raw features, one feature-weight set, softmax tau={core['tau']}, "
             f"batch={core['batch_size']}, at most {core['epochs']} epochs, patience={core['patience']}. "
             "Each arm selects its checkpoint by validation loss. Both training and "
             "evaluation use that arm's mask setting. CPU, one torch thread, deterministic algorithms.", "",
             "Regression data generation/loading and standardization use `model.regression_workflow`; "
             "training uses the T1 `CoreConfig` path with `task_type=regression`. Errors are RMSE in "
             "training-standardized target units (regression), or 1 - accuracy (classification). Lower is better. "
             "SD is variation across three seeds, not a confidence interval.", "",
             "| Dataset | Metric | Mask on, mean +/- SD | All cases, mean +/- SD | Relative change in mean error | All-case better / worse / tied seeds |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for x in summary:
        metric = "RMSE" if x["task_type"] == "regression" else "Error rate"
        change = f"{x['relative_error_change_percent']:+.1f}%" if x["relative_error_change_percent"] is not None else "undefined (zero baseline)"
        lines.append(f"| {x['dataset']} | {metric} | {x['masked_mean_error']:.4f} +/- {x['masked_sd_error']:.4f} | "
                     f"{x['all_case_mean_error']:.4f} +/- {x['all_case_sd_error']:.4f} | "
                     f"{change} | "
                     f"{x['unmasked_better_pairs']} / {x['unmasked_worse_pairs']} / {x['ties']} |")
    lines += ["", f"All-case softmax has lower mean test error on {better} of {dataset_count} datasets. "
              "Negative relative change favors all-case softmax; it is computed from unrounded means.", "",
              f"Training-budget caveat: {cap_count} of {len(unmasked)} unmasked runs select the final allowed "
              f"epoch ({core['epochs']}). This does not establish convergence or each arm's fully optimized performance.", "",
              "![Paired held-out errors](comparison.png)", "",
              "Thin lines connect the same seed. Diamonds show means. Panels have separate vertical scales. "
              f"{seed_count} seeds and one common configuration do not establish a universal preference or replicate "
              "the historical tuned runs. No source default was changed.", "",
              "Reproduce from the repository root:", "", "```powershell",
              f".venv/Scripts/python.exe tools/t1_pre_topk_pilot.py --config {Path(config_path).as_posix()} --out <new-output-directory>", "```", "",
              f"Configuration: [experiment YAML](../../../../{Path(config_path).as_posix()}). "
              "Inspect [all run records](runs.json), [per-seed receipts](runs/), and "
              "[manifest](manifest.json) for source hashes, actual package versions, initialization, "
              "predictions, epochs, runtime, and neighborhood-support checks."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = 2 if dataset_count <= 4 else 3
    nrows = (dataset_count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.25 * nrows), constrained_layout=True, squeeze=False)
    for ax, x in zip(axes.flat, summary):
        for seed in study["seeds"]:
            pair = [next(r["test_error"] for r in rows if r["dataset"] == x["dataset"] and r["seed"] == seed and r["masked"] == flag)
                    for flag in [True, False]]
            ax.plot([0, 1], pair, color="#b8b8b8", linewidth=1, zorder=1)
            ax.scatter([0, 1], pair, c=["#245e8b", "#bb7619"], s=35, zorder=2)
        means = [x["masked_mean_error"], x["all_case_mean_error"]]
        ax.scatter([0, 1], means, marker="D", s=65, c=["#245e8b", "#bb7619"], edgecolor="black", zorder=3)
        ax.set_xticks([0, 1], [f"Top-{core['top_k']} mask", "All-case softmax"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(bottom=0)
        ax.set_title(x["dataset"].replace("_", " ").title())
        ax.set_ylabel("Standardized-target RMSE" if x["task_type"] == "regression" else "Classification error rate")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[dataset_count:]:
        ax.set_visible(False)
    fig.suptitle(f"Pre-Top-K masking: {seed_count} paired seeds per dataset\nLower error is better; diamonds are means")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/t1/pre_topk_pilot_20261030.yaml")
    ap.add_argument("--out", default="results/t1/p6_confirmation_revision/pre_topk_pilot_20261030")
    args = ap.parse_args()
    study = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit("Output already contains a run; use a new --out to preserve its evidence.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(out / "manifest.json", {"study": study, "git": git_state(REPO), "source_sha256": source_fingerprint(REPO),
                                       "environment": environment_info(), "python_executable": sys.executable,
                                       "note": "Actual local packages recorded above; no uv.lock environment-reproduction claim."})
    rows = []
    for ds in study["datasets"]:
        for seed in study["seeds"]:
            data = data_for(ds, seed, study["splits"])
            initial = None
            for masked in [True, False]:
                cfg = CoreConfig(task_type=ds["task_type"], seed=seed, pre_topk_mask=masked, **study["core"])
                model = build_model(data.X_train, data.y_train, cfg, data.num_classes)
                digest = state_digest(model)
                if initial is None:
                    initial = digest
                assert digest == initial, "Paired initialization mismatch"
                assert model.t1_bias_init["k"] == cfg.bias_init_k
                assert model.nn_cdh is None
                initial_train_support, initial_test_support = support(model, data, True), support(model, data, False)
                if masked:
                    assert initial_train_support["positive_weights_max"] <= cfg.top_k
                    assert initial_test_support["positive_weights_max"] <= cfg.top_k
                else:
                    assert initial_train_support["positive_weights_mean"] > cfg.top_k
                    assert initial_test_support["positive_weights_mean"] > cfg.top_k
                started = time.perf_counter()
                trained = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cfg)
                val, test = evaluate(model, data.X_val, data.y_val), evaluate(model, data.X_test, data.y_test)
                error = float(test["rmse_pre"]) if ds["task_type"] == "regression" else 1 - float(test["accuracy_pre"])
                prediction = test["pred_pre"].cpu().tolist()
                row = {"dataset": ds["name"], "task_type": ds["task_type"], "seed": seed, "masked": masked,
                       "core": dataclasses.asdict(cfg), "initial_state_sha256": digest, "bias_initialization": model.t1_bias_init,
                       "initial_train_support": initial_train_support, "initial_test_support": initial_test_support,
                       "final_test_support": support(model, data, False), "best_epoch": trained.best_epoch,
                       "epochs_run": trained.epochs_run, "seconds": time.perf_counter() - started,
                       "validation_loss": val["loss_pre"], "test_error": error, "data": data.describe(),
                       "test_prediction": prediction, "test_target": data.y_test.cpu().tolist(), "history": trained.history}
                if data.y_scale is not None:
                    row["test_rmse_original_units"] = error * data.y_scale[1]
                write_json(out / "runs" / ds["name"] / f"s{seed}" / ("masked.json" if masked else "all_cases.json"), row)
                rows.append(row)
                write_json(out / "runs.json", rows)
                print(f"{ds['name']} seed={seed} masked={masked} error={error:.6f} "
                      f"best_epoch={trained.best_epoch} bias_init_k={cfg.bias_init_k}", flush=True)
    summarize(study, rows, out, args.config)
    print(f"Completed {len(rows)} fits. Report: {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
