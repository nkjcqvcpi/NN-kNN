"""Resumable Cartesian HPO. Candidates never receive the held-out test tensors.

Screen every candidate on development validation streams, refit the declared
number of finalists, freeze a winner, then evaluate it on fresh seeds. This is
a matched local comparison, not certification against a live leaderboard.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import itertools
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from model.common.artifacts import config_id, environment_info, git_state, source_fingerprint, write_json
from model.common.core import CoreConfig, build_model, train_retrieval
from model.t1.data import make_splits


def expand_grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def rank_candidates(rows, ids, seeds):
    ranked = []
    for cid in ids:
        values = [r for r in rows if r["candidate_id"] == cid and r["status"] == "completed"]
        if sorted(r["seed"] for r in values) != sorted(seeds):
            continue
        if not all(math.isfinite(r["validation_error"]) for r in values):
            continue
        ranked.append((float(np.mean([r["validation_error"] for r in values])),
                       float(np.mean([r["validation_loss"] for r in values])), cid))
    return [cid for _, _, cid in sorted(ranked)]


def numeric_metrics(prediction, y, task, scale=None):
    if task == "classification":
        p = np.asarray(prediction, dtype=np.float64)
        p = np.clip(p, 1e-12, None)
        p /= p.sum(1, keepdims=True)
        result = dict(accuracy=float(accuracy_score(y, p.argmax(1))),
                      loss=float(log_loss(y, p, labels=np.arange(p.shape[1]))))
        if p.shape[1] == 2:
            result["roc_auc"] = float(roc_auc_score(y, p[:, 1]))
        return result
    rmse = float(np.sqrt(mean_squared_error(y, np.asarray(prediction).reshape(-1))))
    return dict(rmse_standardized=rmse, rmse_original=rmse * (scale[1] if scale else 1.), loss=rmse**2)


def prediction(model, family, X, task):
    if family in {"knn", "catboost"}:
        return model.predict_proba(X.numpy()) if task == "classification" else model.predict(X.numpy()).reshape(-1)
    model.eval()
    parts = []
    with torch.no_grad():
        for x in X.split(128):
            out = model(x, exclude_identical=False)[0] if family == "nnknn" else model(x)
            if family == "tabm":
                out = out.softmax(-1).mean(1) if task == "classification" else out.mean(1)
            elif family == "mlp" and task == "classification":
                out = out.softmax(-1)
            parts.append(out.detach().cpu())
    return torch.cat(parts).numpy()


def fit_candidate(family, params, data, seed, epochs, patience):
    # data intentionally contains train/validation only.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if family == "nnknn":
        cc = CoreConfig(task_type=data["task"], seed=seed, epochs=epochs, patience=patience,
            batch_size=64, lr_glocal=params["lr_case"], lr_feature=params["lr_case"] * .1, **params)
        model = build_model(data["X_train"], data["y_train"], cc, data["classes"])
        tr = train_retrieval(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"], cc)
        return model, dict(core=dataclasses.asdict(cc), best_epoch=tr.best_epoch, epochs_run=tr.epochs_run,
            optimizer_state=tr.optimizer.state_dict(), history=tr.history)
    if family == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        ctor = KNeighborsClassifier if data["task"] == "classification" else KNeighborsRegressor
        model = ctor(**(params | {"n_neighbors": min(params["n_neighbors"], len(data["y_train"]))}), n_jobs=1)
        model.fit(data["X_train"].numpy(), data["y_train"].numpy())
        return model, {}
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        ctor = CatBoostClassifier if data["task"] == "classification" else CatBoostRegressor
        model = ctor(**params, random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
        model.fit(data["X_train"].numpy(), data["y_train"].numpy(),
            eval_set=(data["X_val"].numpy(), data["y_val"].numpy()), early_stopping_rounds=patience)
        return model, dict(best_epoch=model.get_best_iteration())
    output = data["classes"] if data["task"] == "classification" else 1
    if family == "tabm":
        from tabm import TabM
        model = TabM.make(n_num_features=data["X_train"].shape[1], d_out=output,
            **{k: v for k, v in params.items() if k not in {"lr", "weight_decay"}})
    else:
        layers = []
        dim = data["X_train"].shape[1]
        for _ in range(params["layers"]):
            layers.extend([torch.nn.Linear(dim, params["width"]), torch.nn.ReLU(), torch.nn.Dropout(params["dropout"])])
            dim = params["width"]
        model = torch.nn.Sequential(*layers, torch.nn.Linear(dim, output))
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best, bad, history = (float("inf"), None, -1), 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(data["y_train"]), generator=gen)
        for b in perm.split(64):
            out = model(data["X_train"][b])
            y = data["y_train"][b]
            if family == "tabm":
                # TabM's heads must each learn; averaging here is prohibited.
                target = y[:, None].expand(-1, out.shape[1])
                loss = F.cross_entropy(out.flatten(0, 1), target.flatten()) if output > 1 else F.mse_loss(out[..., 0], target)
            else:
                loss = F.cross_entropy(out, y) if output > 1 else F.mse_loss(out[:, 0], y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural baseline loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = numeric_metrics(prediction(model, family, data["X_val"], data["task"]), data["y_val"].numpy(), data["task"])
        history.append(dict(epoch=epoch, validation_loss=val["loss"]))
        if val["loss"] < best[0] - 1e-9:
            best, bad = (val["loss"], copy.deepcopy(model.state_dict()), epoch), 0
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best[1])
    return model, dict(best_epoch=best[2], epochs_run=epoch, history=history)


def tensorize(value):
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, dict):
        return {k: tensorize(v) for k, v in value.items()}
    return value


def dataset_family(study, ds, family, out):
    out = Path(out)
    job = out / (ds["task_type"] + "__" + ds["name"]) / family
    job.mkdir(parents=True, exist_ok=True)
    if (job / "completed.json").exists():
        return json.loads((job / "completed.json").read_text())
    torch.set_num_threads(study["torch_threads"])
    grid = {config_id(p): p for p in expand_grid(study["grids"][family])}
    rows = []
    data_cache = {}
    def data(seed):
        if seed not in data_cache:
            bundle = make_splits(ds["name"], ds["task_type"], seed, **study["splits"])
            data_cache[seed] = bundle
        return data_cache[seed]
    with (job / "training.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        def trial(stage, cid, seed, epochs, patience, test=False):
            path = job / stage / cid / f"s{seed}"
            path.mkdir(parents=True, exist_ok=True)
            receipt = path / "receipt.json"
            if receipt.exists():
                return json.loads(receipt.read_text())
            started = time.perf_counter()
            try:
                full = data(seed)
                allowed = dict(X_train=full.X_train, y_train=full.y_train, X_val=full.X_val,
                               y_val=full.y_val, task=full.task_type, classes=full.num_classes)
                model, info = fit_candidate(family, grid[cid], allowed, seed, epochs, patience)
                val_prediction = prediction(model, family, full.X_val, full.task_type)
                metrics = numeric_metrics(val_prediction, full.y_val.numpy(), full.task_type, full.y_scale)
                row = dict(status="completed", family=family, candidate_id=cid, parameters=grid[cid],
                    seed=seed, stage=stage, validation_error=1.-metrics["accuracy"] if full.num_classes else metrics["rmse_standardized"],
                    validation_loss=metrics["loss"], validation_metrics=metrics, test_stream_evaluated=test,
                    training_seconds=time.perf_counter()-started, source_sha256=study["source_sha256"],
                    dataset=full.describe())
                if test:
                    # Winner was frozen before this branch can execute.
                    inference_started = time.perf_counter()
                    pred = prediction(model, family, full.X_test, full.task_type)
                    row["inference_seconds"] = time.perf_counter() - inference_started
                    row["inference_queries"] = len(full.y_test)
                    row["training_case_count"] = len(full.y_train)
                    row["active_case_count"] = model.case_count() if family == "nnknn" else len(full.y_train) if family == "knn" else None
                    row["test_metrics"] = numeric_metrics(pred, full.y_test.numpy(), full.task_type, full.y_scale)
                    torch.save(dict(prediction=torch.as_tensor(pred), target=full.y_test), path / "test_predictions.pt")
                    torch.save(tensorize(dataclasses.asdict(full)), path / "data.pt")
                    if family in {"knn", "catboost"}:
                        joblib.dump(model, path / "model.joblib")
                    else:
                        torch.save(dict(model_state=model.state_dict(), info=info), path / "model.pt")
                torch.save(dict(prediction=torch.as_tensor(val_prediction), target=full.y_val), path / "validation_predictions.pt")
                write_json(receipt, row)
                print(json.dumps(dict(stage=stage, seed=seed, candidate=cid, error=row["validation_error"])), flush=True)
                return row
            except Exception:
                failure = dict(status="failed", stage=stage, candidate_id=cid, seed=seed, error=traceback.format_exc())
                write_json(path / "failure.json", failure)
                print(failure["error"], flush=True)
                return failure
        for cid in grid:
            for seed in study["development_seeds"]:
                rows.append(trial("screen", cid, seed, study["screen_epochs"], study["screen_patience"]))
        ranked = rank_candidates(rows, grid, study["development_seeds"])
        finalists = ranked[:study["finalists"]]
        write_json(job / "shortlist.json", dict(candidate_ids=finalists, selection_stream="development_validation_only"))
        final_rows = [trial("refit", cid, seed, study["refit_epochs"], study["refit_patience"])
                      for cid in finalists for seed in study["development_seeds"]]
        ranked = rank_candidates(final_rows, finalists, study["development_seeds"])
        if not ranked:
            write_json(job / "failed.json", dict(reason="No candidate completed on every development seed"))
            return dict(dataset=ds, family=family, status="failed")
        winner = ranked[0]
        write_json(job / "selection.json", dict(candidate_id=winner, parameters=grid[winner],
            development_seeds=study["development_seeds"], test_stream_used=False,
            source_sha256=study["source_sha256"]))
        evaluations = [trial("evaluation", winner, seed, study["refit_epochs"], study["refit_patience"], test=True)
                       for seed in study["evaluation_seeds"]]
        complete = all(r["status"] == "completed" for r in rows + final_rows + evaluations)
        result = dict(dataset=ds, family=family, status="completed" if complete else "partial",
            winner=winner, parameters=grid[winner], screen_successes=sum(r["status"] == "completed" for r in rows),
            screen_expected=len(grid)*len(study["development_seeds"]), evaluation=evaluations)
        write_json(job / ("completed.json" if result["status"] == "completed" else "failed.json"), result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-dataset")
    parser.add_argument("--only-family")
    args = parser.parse_args()
    study = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint(REPO)
    frozen = out / "study.json"
    if frozen.exists():
        old = json.loads(frozen.read_text())
        if old["source_sha256"] != source or old["configuration_id"] != config_id(study):
            raise ValueError("Source or configuration changed; use a fresh output directory")
    else:
        write_json(frozen, dict(configuration=study, configuration_id=config_id(study), source_sha256=source,
            environment={**environment_info(), "baseline_packages": {
                p: importlib.metadata.version(p) for p in ("catboost", "openml", "tabm", "scikit-learn")}}, git=git_state(REPO), test_based_selection=False,
            expected_jobs=len(study["datasets"])*len(study["families"])))
        with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for directory in ("model", "tools", "configs", "tests", "datasets"):
                for path in sorted((REPO / directory).rglob("*")):
                    if path.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in path.parts:
                        z.write(path, path.relative_to(REPO).as_posix())
    study["source_sha256"] = source
    tasks = [(ds, fam) for ds in study["datasets"] for fam in study["families"]
             if (not args.only_dataset or ds["name"] == args.only_dataset)
             and (not args.only_family or fam == args.only_family)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(dataset_family, study, ds, fam, str(out)): (ds, fam) for ds, fam in tasks}
        for future in as_completed(futures):
            ds, fam = futures[future]
            try:
                result = future.result()
            except Exception:
                result = dict(dataset=ds, family=fam, status="failed", error=traceback.format_exc())
            results.append(result)
            write_json(out / "progress.json", dict(expected_jobs=len(tasks), finished_jobs=len(results),
                complete=sum(r["status"] == "completed" for r in results), failures=[r for r in results if r["status"] != "completed"]))
            print(f"[{len(results)}/{len(tasks)}] {ds['name']} {fam}: {result['status']}", flush=True)
    write_json(out / "results.json", results)
    if any(r["status"] != "completed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
