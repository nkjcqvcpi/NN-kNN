"""Resumable Cartesian HPO. Candidates never receive the held-out test tensors.

Screen every candidate on development validation streams, refit the declared
number of finalists, freeze a winner, then evaluate it on fresh seeds. This is
a matched local comparison, not certification against a live leaderboard.

    python tools/t1_grid_search.py CONFIG --study S --out DIR [--workers N]
    python tools/t1_grid_search.py evaluate --grid DIR --families F.. --seeds S.. --study S --out DIR2
        [--datasets TASK_KEY..] [--workers N]

``evaluate`` re-fits each frozen winner (``selection.json``) of an existing grid
on further seeds through the grid's own evaluation stage; it never re-selects.
Its --out must be new, empty or an earlier evaluate output of the same grid.
NNKNN_DETERMINISTIC=1 turns on ``torch.use_deterministic_algorithms`` in workers
(and sets CUBLAS_WORKSPACE_CONFIG as tools/t1_run.py); an output directory is
resumed only under the setting it was started with. A grid or evaluate output resumes under any code state (owner's
order 2026-11-23: same configuration, not same code); study.json and evaluate.json keep the source fingerprints as
provenance, and ``--out`` is refused only for another configuration, grid or study.

``--study S`` (both forms, required) records every trial as the run unit
``NN-KNN/runs/t1/S/<task>/<family>/<stage>/<candidate>/s<seed>/``; no trial file is written. Record
``trial`` (the receipt row), ``final/*`` numbers, arrays ``validation_prediction``/``validation_target`` (and on
the test stream ``test_prediction``/``test_target``), named states ``model`` (a joblib model as its bytes) and ``data``.
A unit complete with the same trial config (see ``trial_config``: data definition and runtime included; the source
fingerprint is provenance, recorded in the trial row and the unit's session header, not part of the config) is
read back and never fitted again, one complete with another config raises ConfigMismatchError; a failed trial is
marked failed in its unit and refitted on the next run. ``DIR`` holds the job and grid summaries (shortlist,
selection, completed, evaluation, results, progress, study.json with ``"study": S``, training.log). The
``evaluation`` entries of completed.json and evaluation.json only name their trials (``records.pointer``: status,
stage, candidate id, seed); every number is read from the trial unit (``records.trial_row``).
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
from functools import lru_cache
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
from model.common.tracking import open_run, resolve_unit, load_run
from model.t1 import records
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


def nnknn_core_settings(params):
    """CoreConfig fields of an NN-kNN candidate exactly as this grid trains it.

    Experiments that reuse a selected candidate must rebuild it with these
    settings: the selection is only valid for them (e.g. the batch size).
    """
    return {**params, "batch_size": 64, "lr_glocal": params["lr_case"], "lr_feature": params["lr_case"] * .1}


def fit_candidate(family, params, data, seed, epochs, patience, *, run=None):
    # data intentionally contains train/validation only.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if family == "nnknn":
        cc = CoreConfig(task_type=data["task"], seed=seed, epochs=epochs, patience=patience,
                        **nnknn_core_settings(params))
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
    start, stopped = 1, False
    if run is not None:
        if getattr(run, "run_id", None) is None:
            raise TypeError("an open recorded run is required")
        resumed = run.load_resume()
        if resumed is not None:
            state = resumed.state
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            best, bad, history = state["best"], state["bad"], state["history"]
            start, stopped = state["epoch"]+1, state["stopped"]
            resumed.restore_rng({"shuffle":gen})
    epoch = start-1
    for epoch in range(start, epochs + 1) if not stopped else ():
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
        stopped = bad > patience
        if run is not None:
            run.log_step(epoch, {"validation_loss":val["loss"]})
            run.save_resume(records.storable(dict(epoch=epoch, model=model.state_dict(),
                optimizer=opt.state_dict(), best=best, bad=bad, history=history, stopped=stopped)),
                step=epoch, generators={"shuffle":gen})
        if stopped:
            break
    model.load_state_dict(best[1])
    return model, dict(best_epoch=best[2], epochs_run=epoch, history=history)


def tensorize(value):
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, dict):
        return {k: tensorize(v) for k, v in value.items()}
    return value


def task_key(ds):
    return ds["task_type"] + "__" + ds["name"]


def dataset_splits(study, ds):
    """make_splits options for one dataset entry; its own max_train (null = uncapped) overrides the study's."""
    splits = dict(study["splits"])
    if "max_train" in ds:
        splits["max_train"] = ds["max_train"]
    return splits


def split_loader(study, ds):
    """Memoized seed -> split bundle of one dataset entry.

    The loader carries its data definition (``definition``: the dataset entry and the make_splits options), which
    a recorded trial puts into its unit config, so that a unit is only reused for the same data.
    """
    cache = {}
    splits = dataset_splits(study, ds)
    def data(seed):
        if seed not in cache:
            cache[seed] = make_splits(ds["name"], ds["task_type"], seed, **splits)
        return cache[seed]
    data.definition = {"dataset": dict(ds), "splits": splits}
    return data


def data_definition(data):
    """The data definition a recorded trial needs (see split_loader); a loader without one cannot be recorded."""
    definition = getattr(data, "definition", None)
    if definition is None:
        raise ValueError("a recorded trial needs a split_loader data loader (its dataset entry and splits)")
    return records.jsonable(definition)


def trial_runtime(torch_threads=None):
    """Settings outside the trial's parameters that can change its numbers: torch threads (this process's, unless
    given) and deterministic kernels."""
    return {"torch_threads": torch.get_num_threads() if torch_threads is None else int(torch_threads),
            "deterministic_algorithms": deterministic_requested()}


def deterministic_requested():
    return os.environ.get("NNKNN_DETERMINISTIC") == "1"


def apply_determinism():
    """NNKNN_DETERMINISTIC=1 forces deterministic torch kernels in this process, as tools/t1_run.py."""
    if deterministic_requested():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS
        torch.use_deterministic_algorithms(True)


def split_digest(indices):
    """sha256 of a split_indices mapping as canonical JSON; also recomputable from older receipts' lists."""
    return hashlib.sha256(json.dumps(indices, sort_keys=True, separators=(",", ":"), default=int).encode()).hexdigest()


def compact_row(row):
    """Per-trial JSON stores split_indices_sha256 instead of the index lists (hundreds of KB on large tasks).

    Receipts written before this change still hold the lists; they are compacted when reused.
    """
    info = row.get("dataset")
    if not isinstance(info, dict) or "split_indices" not in info:
        return row
    info = {("split_indices_sha256" if k == "split_indices" else k): split_digest(v) if k == "split_indices" else v
            for k, v in info.items()}
    return {**row, "dataset": info}


class TrialUnit:
    """A trial's outputs recorded into its run unit."""

    def __init__(self, run):
        self.run = run

    def predictions(self, name, prediction, target):
        self.run.log_array(f"{name}_prediction", torch.as_tensor(prediction).numpy())
        self.run.log_array(f"{name}_target", torch.as_tensor(target).numpy())

    def data(self, obj):
        self.run.save_state("data", records.storable(obj))

    def model(self, model, family, info):
        if family in {"knn", "catboost"}:
            import io
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            blob = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
            self.run.save_state("model", {"format": "joblib", "bytes": torch.from_numpy(blob)})
        else:
            self.run.save_state("model", records.storable(dict(model_state=model.state_dict(), info=info)))

    def receipt(self, row):
        final = {k: v for k, v in row.items() if k in ("validation_error", "validation_loss")}
        final.update({f"validation/{k}": v for k, v in (row.get("validation_metrics") or {}).items()})
        final.update({f"test/{k}": v for k, v in (row.get("test_metrics") or {}).items()})
        self.run.log_final(records.numeric(final))
        self.run.log_record("trial", records.ordered(row))

    def failure(self, failure, exc):
        self.run.fail(exc)


def trial_model(path):
    """The fitted model of a recorded trial unit: a joblib model, or the ``{model_state, info}`` dict of a torch one."""
    from model.common.tracking import load_state
    state = load_state(path, "model")
    if isinstance(state, dict) and state.get("format") == "joblib":
        import io
        return joblib.load(io.BytesIO(state["bytes"].numpy().tobytes()))
    return state


@lru_cache(maxsize=None)
def current_source():
    """Source fingerprint of this checkout (provenance of the trials this process fits)."""
    return source_fingerprint(REPO)


def trial_config(job, family, cid, params, stage, data, seed, epochs, patience, test=False, torch_threads=None):
    """Config of a recorded trial unit: the configuration its numbers depend on. The data definition (dataset entry
    and splits) and the runtime settings are part of it, so a study shared by several grids or evaluate runs reuses a
    trial only when it is configured the same way (else ConfigMismatchError). The source fingerprint is not (owner's
    order 2026-11-23): units recorded with ``source_sha256`` in their config match through tracking.config_identity."""
    return dict(kind="t1_grid_trial", task=Path(job).parent.name, family=family, candidate_id=cid,
                parameters=records.jsonable(params), stage=stage, seed=seed, epochs=epochs, patience=patience,
                test=test, data=data_definition(data), runtime=trial_runtime(torch_threads))


def run_trial(job, family, cid, params, stage, data, seed, epochs, patience, test=False, *, study):
    """One fit of candidate ``cid``, recorded as the run unit ``t1/<study>/<task>/<family>/<stage>/<cid>/s<seed>``
    (task and family are the job directory's names). ``data(seed)`` gives the splits. A unit complete with the same
    ``trial_config`` is read back before anything is fitted (another config raises ConfigMismatchError).
    """
    if not study:
        raise TypeError("run_trial needs a study: every grid trial is recorded as a run unit")
    key = records.key(study, Path(job).parent.name, Path(job).name, stage, cid, f"s{seed}")
    config = trial_config(job, family, cid, params, stage, data, seed, epochs, patience, test)
    info = resolve_unit(key, config=config)
    if info is not None:
        return compact_row(records.unordered(load_run(info.path, arrays=False).record("trial")))
    run = open_run(key, config=config)
    row = fit_trial(TrialUnit(run), family, cid, params, stage, data, seed, epochs, patience, test)
    if row["status"] == "completed":
        run.complete()
    return row


def fit_trial(out, family, cid, params, stage, data, seed, epochs, patience, test):
    """Fit, evaluate and record one trial through ``out`` (a TrialUnit); returns its row or failure."""
    started = time.perf_counter()
    try:
        full = data(seed)
        allowed = dict(X_train=full.X_train, y_train=full.y_train, X_val=full.X_val,
                       y_val=full.y_val, task=full.task_type, classes=full.num_classes)
        model, info = fit_candidate(family, params, allowed, seed, epochs, patience)
        val_prediction = prediction(model, family, full.X_val, full.task_type)
        metrics = numeric_metrics(val_prediction, full.y_val.numpy(), full.task_type, full.y_scale)
        row = compact_row(dict(status="completed", family=family, candidate_id=cid, parameters=params,
            seed=seed, stage=stage, validation_error=1.-metrics["accuracy"] if full.num_classes else metrics["rmse_standardized"],
            validation_loss=metrics["loss"], validation_metrics=metrics, test_stream_evaluated=test,
            training_seconds=time.perf_counter()-started, source_sha256=current_source(),
            dataset=full.describe()))
        if torch.are_deterministic_algorithms_enabled():
            row["deterministic_algorithms"] = True
        if test:
            # Winner was frozen before this branch can execute.
            inference_started = time.perf_counter()
            pred = prediction(model, family, full.X_test, full.task_type)
            row["inference_seconds"] = time.perf_counter() - inference_started
            row["inference_queries"] = len(full.y_test)
            row["training_case_count"] = len(full.y_train)
            row["active_case_count"] = model.case_count() if family == "nnknn" else len(full.y_train) if family == "knn" else None
            row["test_metrics"] = numeric_metrics(pred, full.y_test.numpy(), full.task_type, full.y_scale)
            out.predictions("test", pred, full.y_test)
            out.data(tensorize(dataclasses.asdict(full)))
            out.model(model, family, info)
        out.predictions("validation", val_prediction, full.y_val)
        out.receipt(row)
        print(json.dumps(dict(stage=stage, seed=seed, candidate=cid, error=row["validation_error"])), flush=True)
        return row
    except Exception as exc:
        failure = dict(status="failed", stage=stage, candidate_id=cid, seed=seed, error=traceback.format_exc())
        out.failure(failure, exc)
        print(failure["error"], flush=True)
        return failure


def dataset_family(study, ds, family, out, run_study):
    out = Path(out)
    job = out / task_key(ds) / family
    job.mkdir(parents=True, exist_ok=True)
    if (job / "completed.json").exists():
        return json.loads((job / "completed.json").read_text())
    apply_determinism()
    torch.set_num_threads(study["torch_threads"])
    grid = {config_id(p): p for p in expand_grid(study["grids"][family])}
    rows = []
    data = split_loader(study, ds)
    with (job / "training.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        def trial(stage, cid, seed, epochs, patience, test=False):
            return run_trial(job, family, cid, grid[cid], stage, data, seed, epochs, patience, test, study=run_study)
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
            screen_expected=len(grid)*len(study["development_seeds"]),
            evaluation=[records.pointer(r) for r in evaluations])  # the numbers are in the trial units only
        write_json(job / ("completed.json" if result["status"] == "completed" else "failed.json"), result)
        return result


def evaluate_family(grid_dir, study, ds, family, seeds, out, run_study):
    """Re-fit one task's frozen grid winner on ``seeds`` through the grid's evaluation stage.

    The winner comes from the grid's selection.json and must be a candidate of the
    frozen grid; nothing is screened, refitted or ranked. Seeds already evaluated in
    ``out`` stay in evaluation.json, whose entries (``records.pointer``: status, stage, candidate, seed) name the
    trial units, as completed.json's do.
    """
    key = task_key(ds)
    selection_path = Path(grid_dir) / key / family / "selection.json"
    if not selection_path.exists():
        return dict(dataset=ds, family=family, status="no_selection")
    selection = json.loads(selection_path.read_text())
    grid = {config_id(p): p for p in expand_grid(study["grids"][family])}
    winner = selection["candidate_id"]
    if grid.get(winner) != selection["parameters"]:
        raise ValueError(f"{selection_path}: winner is not a candidate of the frozen grid")
    job = Path(out) / key / family
    job.mkdir(parents=True, exist_ok=True)
    summary = job / "evaluation.json"
    previous = json.loads(summary.read_text())["evaluation"] if summary.exists() else []
    seeds = list(dict.fromkeys([r["seed"] for r in previous] + list(seeds)))
    apply_determinism()
    torch.set_num_threads(study["torch_threads"])
    data = split_loader(study, ds)
    with (job / "training.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        evaluations = [run_trial(job, family, winner, grid[winner], "evaluation", data, seed, study["refit_epochs"],
                                 study["refit_patience"], test=True, study=run_study)
                       for seed in seeds]
    complete = all(r["status"] == "completed" for r in evaluations)
    result = dict(dataset=ds, family=family, status="completed" if complete else "partial", winner=winner,
        parameters=grid[winner], selection=str(selection_path), development_seeds=selection["development_seeds"],
        grid_source_sha256=study["grid_source_sha256"], source_sha256=study["source_sha256"],
        evaluation=[records.pointer(r) for r in evaluations])
    write_json(summary, result)
    return result


def check_run_study(parser, run_study):
    try:  # the run root and the study name, before any trial runs
        records.check_study(run_study)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(f"--study: {exc}")


def check_trial_paths(parser, run_study, tasks, seeds):
    """Refuse, before any trial runs, a study whose deepest trial unit (stage ``evaluation``, a 16-character
    candidate id, the longest seed) would have too long an event file path."""
    if tasks and seeds:
        seed = f"s{max(seeds, key=lambda s: len(str(s)))}"
        try:
            records.check_unit_paths(run_study, [(task_key(ds), fam, "evaluation", "0" * 16, seed) for ds, fam in tasks])
        except ValueError as exc:
            parser.error(f"--study: {exc}")


def baseline_environment():
    return {**environment_info(), "baseline_packages": {
        p: importlib.metadata.version(p) for p in ("catboost", "openml", "tabm", "scikit-learn")}}


def write_source_zip(target):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for directory in ("model", "tools", "configs", "tests", "datasets"):
            for path in sorted((REPO / directory).rglob("*")):
                if path.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(REPO).as_posix())


def run_jobs(submit, tasks, out, workers, skipped=()):
    """Run (dataset, family) jobs in worker processes; statuses in ``skipped`` are not failures."""
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {submit(pool, ds, fam): (ds, fam) for ds, fam in tasks}
        for future in as_completed(futures):
            ds, fam = futures[future]
            try:
                result = future.result()
            except Exception:
                result = dict(dataset=ds, family=fam, status="failed", error=traceback.format_exc())
            results.append(result)
            write_json(out / "progress.json", dict(expected_jobs=len(tasks), finished_jobs=len(results),
                complete=sum(r["status"] == "completed" for r in results),
                failures=[r for r in results if r["status"] not in ("completed", *skipped)]))
            print(f"[{len(results)}/{len(tasks)}] {ds['name']} {fam}: {result['status']}", flush=True)
    write_json(out / "results.json", results)
    if any(r["status"] not in ("completed", *skipped) for r in results):
        raise SystemExit(1)


def evaluate_main(argv):
    parser = argparse.ArgumentParser(prog="t1_grid_search.py evaluate",
        description="Re-fit an existing grid's frozen winners on further seeds; never re-selects.")
    parser.add_argument("--grid", required=True, help="existing grid output directory (read only)")
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="+", help="task keys <task_type>__<name>; default: every grid task")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--study", required=True, help="every trial is recorded as a run unit under NN-KNN/runs/t1/<study>/")
    args = parser.parse_args(argv)
    check_run_study(parser, args.study)
    grid, out = Path(args.grid).resolve(), Path(args.out).resolve()
    if out == grid or grid in out.parents:
        raise ValueError("--out must lie outside the grid directory")
    frozen = json.loads((grid / "study.json").read_text())
    study = dict(frozen["configuration"])
    datasets = {task_key(ds): ds for ds in study["datasets"]}
    keys, families = list(dict.fromkeys(args.datasets or datasets)), list(dict.fromkeys(args.families))
    unknown = sorted(set(keys) - set(datasets)) + sorted(set(families) - set(study["families"]))
    if unknown:
        raise ValueError(f"Not in the grid's frozen study: {unknown}")
    tasks = [(datasets[key], fam) for key in keys for fam in families]
    check_trial_paths(parser, args.study, tasks, args.seeds)
    record_path = out / "evaluate.json"
    # Receipts found under --out are reused unchecked, so only an evaluate output of this grid may be resumed.
    if not record_path.exists() and out.is_dir() and any(out.iterdir()):
        raise ValueError(f"--out {out} is not empty and holds no evaluate.json; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    source = current_source()
    deterministic = deterministic_requested()
    invocation = dict(families=args.families, seeds=args.seeds, datasets=args.datasets,
                      deterministic_algorithms=deterministic, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      source_sha256=source)
    if record_path.exists():
        record = json.loads(record_path.read_text())
        # the source fingerprint is provenance (each invocation records its own), never a reason to refuse
        if record["grid_configuration_id"] != frozen["configuration_id"] or record["grid"] != str(grid):
            raise ValueError("Grid changed; use a fresh output directory")
        if any(i["deterministic_algorithms"] != deterministic for i in record["invocations"]):
            raise ValueError("NNKNN_DETERMINISTIC differs from earlier invocations; use a fresh output directory")
        if record.get("study") != args.study:
            raise ValueError(f"--study differs from earlier invocations ({record.get('study')!r}); resume with the same one")
        record["invocations"].append(invocation)
    else:
        record = dict(grid=str(grid), grid_configuration_id=frozen["configuration_id"],
            grid_source_sha256=frozen["source_sha256"], source_sha256=source, environment=baseline_environment(),
            git=git_state(REPO), test_based_selection=False, selection_source="grid selection.json, frozen",
            invocations=[invocation], study=args.study)
        write_source_zip(out / "source.zip")
    write_json(record_path, record)
    study.update(source_sha256=record["source_sha256"], grid_source_sha256=frozen["source_sha256"])
    run_jobs(lambda pool, ds, fam: pool.submit(evaluate_family, str(grid), study, ds, fam, args.seeds, str(out), args.study),
             tasks, out, args.workers, skipped=("no_selection",))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["evaluate"]:
        return evaluate_main(argv[1:])
    parser = argparse.ArgumentParser(epilog="Re-evaluate frozen winners: t1_grid_search.py evaluate -h")
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-dataset")
    parser.add_argument("--only-family")
    parser.add_argument("--study", required=True, help="every trial is recorded as a run unit under NN-KNN/runs/t1/<study>/")
    args = parser.parse_args(argv)
    check_run_study(parser, args.study)
    study = yaml.safe_load(Path(args.config).read_text())
    tasks = [(ds, fam) for ds in study["datasets"] for fam in study["families"]
             if (not args.only_dataset or ds["name"] == args.only_dataset)
             and (not args.only_family or fam == args.only_family)]
    check_trial_paths(parser, args.study, tasks, [*study["development_seeds"], *study["evaluation_seeds"]])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = current_source()
    frozen = out / "study.json"
    # Recorded only when on, so default study.json files are unchanged; a resume may not switch it.
    deterministic = dict(deterministic_algorithms=True) if deterministic_requested() else {}
    if frozen_old := frozen.exists():
        old = json.loads(frozen.read_text())
        # same configuration, not same code (owner's order 2026-11-23): another source is provenance only
        if old["configuration_id"] != config_id(study):
            raise ValueError("Configuration changed; use a fresh output directory")
        if old.get("deterministic_algorithms", False) != bool(deterministic):
            raise ValueError("NNKNN_DETERMINISTIC differs from the grid's study.json; resume with the same setting")
        if old.get("study") != args.study:
            raise ValueError(f"--study differs from the grid's study.json ({old.get('study')!r}); resume with the same one")
    else:
        write_json(frozen, dict(configuration=study, configuration_id=config_id(study), source_sha256=source,
            environment=baseline_environment(), git=git_state(REPO), test_based_selection=False,
            expected_jobs=len(study["datasets"])*len(study["families"]), **deterministic, study=args.study))
        write_source_zip(out / "source.zip")
    # the grid's source as recorded (selection.json keeps it); every trial records the source that fitted it
    study["source_sha256"] = old["source_sha256"] if frozen_old else source
    run_jobs(lambda pool, ds, fam: pool.submit(dataset_family, study, ds, fam, str(out), args.study), tasks, out, args.workers)


if __name__ == "__main__":
    main()
