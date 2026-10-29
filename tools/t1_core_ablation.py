"""One-factor ablation of the NN-kNN retrieval core around each task's frozen grid winner.

Why: the pre-registered confirmation (results/t1/p6_confirmation_revision/confirmation_report_20261020)
shows F trailing tuned CatBoost (E3a GMR 1.135) and not non-inferior to TabM, while reuse adds little
over the core (E1 GMR 0.974). The core therefore decides the gap. This tool varies one core factor at a
time (neighbourhood size, temperature, learning rate, which parameters learn, representation, weight
sets, regularisation, budget, normaliser) starting from the grid winner of each task, and evaluates
every variant on the grid's evaluation seeds so that the test errors are comparable with the baseline
rows of the same grid. Exploratory and descriptive: nothing here selects or changes F.

Usage:
  .venv/Scripts/python.exe -u tools/t1_core_ablation.py configs/t1/core_ablation_20261027.yaml \
      --out results/t1/p6_confirmation_revision/core_ablation_20261027 --workers 12
  .venv/Scripts/python.exe tools/t1_core_ablation.py <config> --out <dir> --summarize-only
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")
import numpy as np
import torch
import yaml

from model.common.artifacts import config_id, environment_info, source_fingerprint, write_json
from model.common.core import CoreConfig, build_model, evaluate, train_retrieval
from tools.t1_grid_search import (apply_determinism, compact_row, nnknn_core_settings, numeric_metrics,
                                  prediction, split_loader, task_key)

CORE_FIELDS = {f.name for f in dataclasses.fields(CoreConfig)}


def load_study(path):
    study = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    study["source_sha256"] = source_fingerprint(REPO)
    study["config_path"] = str(path)
    return study


def grid_winner(grid_dir, ds):
    sel = Path(grid_dir) / task_key(ds) / "nnknn" / "selection.json"
    return json.loads(sel.read_text())["parameters"]


def baseline_rows(grid_dir, ds, seeds):
    """Mean test error of each baseline family's frozen winner on ``seeds`` (same splits as this ablation)."""
    out = {}
    for fam in ("nnknn", "knn", "catboost", "mlp", "tabm"):
        p = Path(grid_dir) / task_key(ds) / fam / "completed.json"
        if not p.exists():
            continue
        rows = [r for r in json.loads(p.read_text())["evaluation"] if r.get("status") == "completed" and r["seed"] in seeds]
        errs = {r["seed"]: test_error(r["test_metrics"]) for r in rows}
        if errs:
            out[fam] = dict(per_seed=errs, mean=float(np.mean(list(errs.values()))), parameters=json.loads(p.read_text())["parameters"])
    return out


def test_error(metrics):
    return 1.0 - metrics["accuracy"] if "accuracy" in metrics else metrics["rmse_standardized"]


def resolve_variant(base_params, variant, study):
    """CoreConfig keyword arguments of one variant: grid expansion of the winner, then the override.

    Override values: a plain value replaces the field; the string "base" keeps the expanded base
    value (used to freeze one learning rate while the others keep the grid's ratio). An override that
    sets ``lr_case`` without ``lr_glocal``/``lr_feature`` re-derives those by the grid's ratios.
    """
    expanded = nnknn_core_settings(base_params)
    expanded.setdefault("epochs", study["epochs"])
    expanded.setdefault("patience", study["patience"])
    override = dict(variant.get("override") or {})
    if "lr_case" in override and "lr_glocal" not in override and "lr_feature" not in override and override["lr_case"] != "base":
        override["lr_glocal"] = override["lr_case"]
        override["lr_feature"] = override["lr_case"] * 0.1
    settings = dict(expanded)
    for k, v in override.items():
        if k not in CORE_FIELDS:
            raise ValueError(f"{variant['name']}: {k} is not a CoreConfig field")
        if v == "base":
            continue
        settings[k] = tuple(v) if k == "mlp_dims" else v
    for k, v in override.items():
        if v == "=lr_case":  # copy the resolved case rate (e.g. an encoder trained at the case rate)
            settings[k] = settings["lr_case"]
    if settings.get("mcb_enabled") and settings["representation"] != "mlp":
        settings["representation"] = "mlp"  # MCB needs an encoder; recorded in the receipt
    return settings


@torch.no_grad()
def retrieval_diagnostics(model, X):
    """Effective neighbourhood size exp(H(w)) and the learned feature weights (first set), for interpretation."""
    model.eval()
    ess = []
    for s in range(0, X.size(0), 256):
        w = model.retrieve(X[s:s + 256], exclude_identical=False)["weights"].clamp_min(1e-12)
        ess.append(torch.exp(-(w * w.log()).sum(1)))
    d = dict(effective_neighbours_mean=float(torch.cat(ess).mean()))
    fw = model.glocal_weightor.feature_weights.detach().cpu()
    if fw.shape[1] <= 128:
        d["feature_weights_set0"] = torch.nn.functional.leaky_relu(fw[0], 0.001).tolist()
    d["bias_mean"] = float(model.biases[: model.case_count()].mean())
    d["bias_std"] = float(model.biases[: model.case_count()].std()) if model.case_count() > 1 else 0.0
    return d


def fit_variant(settings, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cc = CoreConfig(task_type=data.task_type, seed=seed, **settings)
    model = build_model(data.X_train, data.y_train, cc, data.num_classes)
    tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc)
    return model, cc, tr


def run_trial(job, variant, settings, data, seed, source):
    path = Path(job) / f"s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    receipt = path / "receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    started = time.perf_counter()
    try:
        full = data(seed)
        model, cc, tr = fit_variant(settings, full, seed)
        training_seconds = time.perf_counter() - started
        val_pred = prediction(model, "nnknn", full.X_val, full.task_type)
        val = numeric_metrics(val_pred, full.y_val.numpy(), full.task_type, full.y_scale)
        t0 = time.perf_counter()
        pred = prediction(model, "nnknn", full.X_test, full.task_type)
        inference_seconds = time.perf_counter() - t0
        test = numeric_metrics(pred, full.y_test.numpy(), full.task_type, full.y_scale)
        train_fit = evaluate(model, full.X_train, full.y_train)  # no LOO: resubstitution, for the fit/generalisation gap
        row = compact_row(dict(status="completed", variant=variant, settings=dataclasses.asdict(cc), seed=seed,
            validation_error=test_error(val), validation_metrics=val, test_error=test_error(test), test_metrics=test,
            train_resubstitution_error=(1.0 - train_fit["accuracy_pre"]) if full.num_classes else train_fit["rmse_pre"],
            best_epoch=tr.best_epoch, epochs_run=tr.epochs_run, best_val_loss=tr.best_val,
            training_seconds=training_seconds, inference_seconds=inference_seconds,
            diagnostics=retrieval_diagnostics(model, full.X_test), source_sha256=source, dataset=full.describe()))
        torch.save(dict(prediction=torch.as_tensor(pred), target=full.y_test), path / "test_predictions.pt")
        write_json(receipt, row)
        print(json.dumps(dict(variant=variant, seed=seed, val=row["validation_error"], test=row["test_error"])), flush=True)
        return row
    except Exception:
        failure = dict(status="failed", variant=variant, seed=seed, error=traceback.format_exc())
        write_json(path / "failure.json", failure)
        print(failure["error"], flush=True)
        return failure


def task_variant(study, ds, variant, out):
    out = Path(out)
    job = out / task_key(ds) / variant["name"]
    job.mkdir(parents=True, exist_ok=True)
    done = job / "completed.json"
    if done.exists():
        return json.loads(done.read_text())
    apply_determinism()
    torch.set_num_threads(study.get("torch_threads", 1))
    base = grid_winner(study["grid"], ds)
    settings = resolve_variant(base, variant, study)
    data = split_loader(study, ds)
    with (job / "training.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        rows = [run_trial(job, variant["name"], settings, data, seed, study["source_sha256"]) for seed in study["evaluation_seeds"]]
    result = dict(task=task_key(ds), variant=variant["name"], override=variant.get("override") or {}, base_parameters=base,
                  settings=settings, status="completed" if all(r["status"] == "completed" for r in rows) else "partial", rows=rows)
    write_json(done if result["status"] == "completed" else job / "partial.json", result)
    return result


def dedupe_variants(study, ds):
    """Variants whose resolved settings equal an earlier variant's are aliases (recorded, not refit)."""
    base = grid_winner(study["grid"], ds)
    seen, keep, aliases = {}, [], {}
    for v in study["variants"]:
        cid = config_id(resolve_variant(base, v, study))
        if cid in seen:
            aliases[v["name"]] = seen[cid]
        else:
            seen[cid] = v["name"]
            keep.append(v)
    return keep, aliases


def gmean_ratio(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    eps = 1e-4
    return float(math.exp(np.mean([math.log((x + eps) / (y + eps)) for x, y in pairs])))


def summarize(study, out):
    out = Path(out)
    seeds = study["evaluation_seeds"]
    tasks = {}
    lines = ["# NN-kNN core ablation (exploratory, descriptive)", "",
             f"Config `{study['config_path']}`; grid winners and baselines from `{study['grid']}`; evaluation seeds {seeds}; "
             f"epochs {study['epochs']}, patience {study['patience']}. Error = 1 - accuracy (classification) or "
             "train-standardised RMSE (regression), mean over seeds; SE = standard error over seeds. "
             "Ratio = geometric mean over seeds of variant / base error (< 1 is better than the task's grid winner). "
             "Nothing here was pre-registered; no variant replaces F.", ""]
    csv = ["task,variant,n_seeds,test_error_mean,test_error_se,val_error_mean,train_resub_error_mean,ratio_to_base,ratio_to_catboost,ratio_to_tabm,best_epoch_mean,effective_neighbours_mean,training_seconds_mean,alias_of"]
    for ds in study["datasets"]:
        key = task_key(ds)
        keep, aliases = dedupe_variants(study, ds)
        base_rows = None
        results = {}
        for v in keep:
            p = out / key / v["name"] / "completed.json"
            if not p.exists():
                p = out / key / v["name"] / "partial.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            rows = [x for x in r["rows"] if x["status"] == "completed"]
            results[v["name"]] = (r, rows)
        if "base" in results:
            base_rows = {x["seed"]: x["test_error"] for x in results["base"][1]}
        bl = baseline_rows(study["grid"], ds, seeds)
        tasks[key] = dict(baselines={k: v["mean"] for k, v in bl.items()}, variants={})
        lines += [f"## {key}", "",
                  "Grid winner: `" + json.dumps(grid_winner(study["grid"], ds)) + "`. Baseline test errors on the same seeds: "
                  + ", ".join(f"{k} {v['mean']:.4f}" for k, v in bl.items()), "",
                  "| variant | override | seeds | test error | SE | val error | train resub. | ratio to base | ratio to CatBoost | ratio to TabM | best epoch | eff. neighbours |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in keep:
            if v["name"] not in results:
                lines.append(f"| {v['name']} | `{json.dumps(v.get('override') or {})}` | 0 | not run | | | | | | | | |")
                continue
            r, rows = results[v["name"]]
            te = [x["test_error"] for x in rows]
            by_seed = {x["seed"]: x["test_error"] for x in rows}
            common = [s for s in seeds if s in by_seed]
            ratio_base = gmean_ratio([by_seed[s] for s in common], [base_rows.get(s) for s in common]) if base_rows else float("nan")
            ratio_cb = gmean_ratio([by_seed[s] for s in common], [bl["catboost"]["per_seed"].get(s) for s in common]) if "catboost" in bl else float("nan")
            ratio_tm = gmean_ratio([by_seed[s] for s in common], [bl["tabm"]["per_seed"].get(s) for s in common]) if "tabm" in bl else float("nan")
            rec = dict(n=len(rows), test_error=float(np.mean(te)), se=float(np.std(te, ddof=1) / math.sqrt(len(te))) if len(te) > 1 else float("nan"),
                       val_error=float(np.mean([x["validation_error"] for x in rows])),
                       train_resub=float(np.mean([x["train_resubstitution_error"] for x in rows])),
                       ratio_to_base=ratio_base, ratio_to_catboost=ratio_cb, ratio_to_tabm=ratio_tm,
                       best_epoch=float(np.mean([x["best_epoch"] for x in rows])),
                       eff_neighbours=float(np.mean([x["diagnostics"]["effective_neighbours_mean"] for x in rows])),
                       training_seconds=float(np.mean([x["training_seconds"] for x in rows])), aliases=[a for a, b in aliases.items() if b == v["name"]])
            tasks[key]["variants"][v["name"]] = rec
            name = v["name"] + (f" (= {', '.join(rec['aliases'])})" if rec["aliases"] else "")
            lines.append(f"| {name} | `{json.dumps(v.get('override') or {})}` | {rec['n']} | {rec['test_error']:.4f} | {rec['se']:.4f} | {rec['val_error']:.4f} | "
                         f"{rec['train_resub']:.4f} | {ratio_base:.3f} | {ratio_cb:.3f} | {ratio_tm:.3f} | {rec['best_epoch']:.0f} | {rec['eff_neighbours']:.1f} |")
            csv.append(",".join(str(x) for x in [key, v["name"], rec["n"], rec["test_error"], rec["se"], rec["val_error"], rec["train_resub"], ratio_base, ratio_cb, ratio_tm,
                                                 rec["best_epoch"], rec["eff_neighbours"], rec["training_seconds"], ""]))
        for a, b in aliases.items():
            csv.append(",".join([key, a, "", "", "", "", "", "", "", "", "", "", "", b]))
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    write_json(out / "summary.json", dict(config=study["config_path"], grid=study["grid"], seeds=seeds, tasks=tasks))
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-dataset", help="dataset name, e.g. zebra")
    parser.add_argument("--only-variant", nargs="+")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    study = load_study(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(study, out)
        print(f"summary written to {out / 'summary.md'}")
        return 0
    write_json(out / "study.json", dict(configuration=study, environment=environment_info()))
    jobs = []
    for ds in study["datasets"]:
        if args.only_dataset and ds["name"] != args.only_dataset:
            continue
        keep, aliases = dedupe_variants(study, ds)
        write_json(out / task_key(ds) / "aliases.json", aliases)
        for v in keep:
            if args.only_variant and v["name"] not in args.only_variant:
                continue
            jobs.append((ds, v))
    # Slow tasks first so that the tail of the run is short.
    order = {d["name"]: i for i, d in enumerate(study["datasets"])}
    jobs.sort(key=lambda j: (-float(j[0].get("cost", 1.0)), order[j[0]["name"]]))
    print(f"{len(jobs)} task/variant jobs x {len(study['evaluation_seeds'])} seeds", flush=True)
    finished, failures = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task_variant, study, ds, v, str(out)): (ds, v) for ds, v in jobs}
        for fut in as_completed(futures):
            ds, v = futures[fut]
            try:
                r = fut.result()
                status = r["status"]
            except Exception:
                status = "failed"
                failures.append(dict(task=task_key(ds), variant=v["name"], error=traceback.format_exc()))
            finished += 1
            print(json.dumps(dict(done=finished, of=len(jobs), task=task_key(ds), variant=v["name"], status=status)), flush=True)
            write_json(out / "progress.json", dict(expected_jobs=len(jobs), finished_jobs=finished, failures=failures))
    summarize(study, out)
    print(f"summary written to {out / 'summary.md'}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
