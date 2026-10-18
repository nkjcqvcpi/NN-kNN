"""Run the pending trials of t1_grid_search.py jobs concurrently; the grid's own run then reuses them.

t1_grid_search.py gives each (dataset, family) job one worker and fits its trials one after
another, so cores sit idle when a few slow jobs remain. This tool fits the same trials with the
grid's own ``run_trial`` (same receipts, paths and source hash), several at a time, stage by stage:
every screen trial; then the refit of the finalists that the grid's ``rank_candidates`` picks from
those receipts; then the evaluation of the winner it picks from the refits. It writes trial
receipts only. Afterwards rerun the grid command unchanged: it reuses every receipt and writes
shortlist.json, selection.json and completed.json itself.

All grid logic is loaded from ``--grid-repo``, the checkout the grid was started from, whose
source fingerprint must equal the grid's study.json. Grids started before ``run_trial`` existed
(e.g. 79dbb31) keep the trial inside ``dataset_family``; ``legacy_trial`` repeats that closure
statement for statement with the checkout's own functions. Stop the grid's own run on these jobs
first: receipts are written without locking.

    python tools/t1_grid_prefill.py --grid-repo CHECKOUT --out GRID_OUT --jobs TASK_KEY/FAMILY.. \
        [--workers N] [--workers-file PATH]

Jobs are served in the order given (put the longest first). ``--workers-file`` holds an integer
that is re-read before each submission, so the concurrency can be raised or lowered while running.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

_MODULES = {}


def grid_module(repo):
    """tools/t1_grid_search.py of ``repo``, imported once per process (it puts ``repo`` first on sys.path)."""
    repo = str(Path(repo).resolve())
    if repo not in _MODULES:
        spec = importlib.util.spec_from_file_location("t1_grid_search_pinned", Path(repo) / "tools" / "t1_grid_search.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[repo] = module
    return _MODULES[repo]


def task_key(ds):
    return ds["task_type"] + "__" + ds["name"]


def deterministic_requested():
    return os.environ.get("NNKNN_DETERMINISTIC") == "1"


def load_study(gs, repo, out):
    """The grid's frozen study, refused unless ``repo`` is the source and setting it was started with."""
    frozen = json.loads((Path(out) / "study.json").read_text())
    study = dict(frozen["configuration"])
    if gs.config_id(study) != frozen["configuration_id"]:
        raise ValueError("study.json configuration does not match its configuration_id")
    if gs.source_fingerprint(Path(repo)) != frozen["source_sha256"]:
        raise ValueError(f"{repo} is not the source the grid was started from; use its pinned checkout")
    if frozen.get("deterministic_algorithms", False) != deterministic_requested():
        raise ValueError("NNKNN_DETERMINISTIC differs from the grid's study.json")
    study["source_sha256"] = frozen["source_sha256"]
    return study


def legacy_trial(gs, study, ds, family, job, cid, params, stage, seed, epochs, patience, test):
    """The ``trial`` closure of dataset_family in grids without ``run_trial`` (79dbb31), statement for statement.

    The grid's data_cache only memoizes make_splits per seed; a fresh call builds the same bundle.
    """
    path = job / stage / cid / f"s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    receipt = path / "receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    torch = gs.torch
    started = time.perf_counter()
    try:
        full = gs.make_splits(ds["name"], ds["task_type"], seed, **study["splits"])
        allowed = dict(X_train=full.X_train, y_train=full.y_train, X_val=full.X_val,
                       y_val=full.y_val, task=full.task_type, classes=full.num_classes)
        model, info = gs.fit_candidate(family, params, allowed, seed, epochs, patience)
        val_prediction = gs.prediction(model, family, full.X_val, full.task_type)
        metrics = gs.numeric_metrics(val_prediction, full.y_val.numpy(), full.task_type, full.y_scale)
        row = dict(status="completed", family=family, candidate_id=cid, parameters=params,
            seed=seed, stage=stage, validation_error=1.-metrics["accuracy"] if full.num_classes else metrics["rmse_standardized"],
            validation_loss=metrics["loss"], validation_metrics=metrics, test_stream_evaluated=test,
            training_seconds=time.perf_counter()-started, source_sha256=study["source_sha256"],
            dataset=full.describe())
        if test:
            # Winner was frozen before this branch can execute.
            inference_started = time.perf_counter()
            pred = gs.prediction(model, family, full.X_test, full.task_type)
            row["inference_seconds"] = time.perf_counter() - inference_started
            row["inference_queries"] = len(full.y_test)
            row["training_case_count"] = len(full.y_train)
            row["active_case_count"] = model.case_count() if family == "nnknn" else len(full.y_train) if family == "knn" else None
            row["test_metrics"] = gs.numeric_metrics(pred, full.y_test.numpy(), full.task_type, full.y_scale)
            torch.save(dict(prediction=torch.as_tensor(pred), target=full.y_test), path / "test_predictions.pt")
            torch.save(gs.tensorize(gs.dataclasses.asdict(full)), path / "data.pt")
            if family in {"knn", "catboost"}:
                gs.joblib.dump(model, path / "model.joblib")
            else:
                torch.save(dict(model_state=model.state_dict(), info=info), path / "model.pt")
        torch.save(dict(prediction=torch.as_tensor(val_prediction), target=full.y_val), path / "validation_predictions.pt")
        gs.write_json(receipt, row)
        print(json.dumps(dict(stage=stage, seed=seed, candidate=cid, error=row["validation_error"])), flush=True)
        return row
    except Exception:
        failure = dict(status="failed", stage=stage, candidate_id=cid, seed=seed, error=gs.traceback.format_exc())
        gs.write_json(path / "failure.json", failure)
        print(failure["error"], flush=True)
        return failure


def fit_trial(repo, study, ds, family, out, stage, cid, seed, test):
    """One grid trial in a worker process, exactly as dataset_family would run it."""
    gs = grid_module(repo)
    if hasattr(gs, "apply_determinism"):
        gs.apply_determinism()
    gs.torch.set_num_threads(study["torch_threads"])
    grid = {gs.config_id(p): p for p in gs.expand_grid(study["grids"][family])}
    epochs, patience = ((study["screen_epochs"], study["screen_patience"]) if stage == "screen"
                        else (study["refit_epochs"], study["refit_patience"]))
    job = Path(out) / task_key(ds) / family
    job.mkdir(parents=True, exist_ok=True)
    with (job / "training.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        if hasattr(gs, "run_trial"):
            row = gs.run_trial(job, family, cid, grid[cid], stage, gs.split_loader(study, ds), seed, epochs, patience,
                               study["source_sha256"], test)
        else:
            row = legacy_trial(gs, study, ds, family, job, cid, grid[cid], stage, seed, epochs, patience, test)
    return row["status"]


def receipt(job, stage, cid, seed):
    path = job / stage / cid / f"s{seed}" / "receipt.json"
    return json.loads(path.read_text()) if path.exists() else None


def next_trials(gs, study, out, ds, family):
    """(stage, [(cid, seed), ..]) still missing at the job's first unfinished stage; ("done", []) at the end.

    Finalists and the winner come from the grid's rank_candidates over the receipts, as in dataset_family.
    """
    job = Path(out) / task_key(ds) / family
    seeds = study["development_seeds"]
    grid = {gs.config_id(p): p for p in gs.expand_grid(study["grids"][family])}
    screen = [(cid, s) for cid in grid for s in seeds]
    rows = [receipt(job, "screen", *t) for t in screen]
    if None in rows:
        return "screen", [t for t, r in zip(screen, rows) if r is None]
    finalists = gs.rank_candidates(rows, grid, seeds)[:study["finalists"]]
    refit = [(cid, s) for cid in finalists for s in seeds]
    rows = [receipt(job, "refit", *t) for t in refit]
    if None in rows:
        return "refit", [t for t, r in zip(refit, rows) if r is None]
    ranked = gs.rank_candidates(rows, finalists, seeds)
    if not ranked:
        return "no_winner", []
    evaluation = [(ranked[0], s) for s in study["evaluation_seeds"]]
    missing = [t for t in evaluation if receipt(job, "evaluation", *t) is None]
    return ("evaluation", missing) if missing else ("done", [])


def concurrency(args):
    if args.workers_file:
        with contextlib.suppress(OSError, ValueError):
            return max(0, int(Path(args.workers_file).read_text().strip()))
    return args.workers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-repo", required=True, help="checkout the grid was started from")
    parser.add_argument("--out", required=True, help="the grid's output directory")
    parser.add_argument("--jobs", nargs="+", required=True, help="TASK_KEY/FAMILY, longest first")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--workers-file")
    parser.add_argument("--max-workers", type=int, default=16, help="process pool size; the active limit is --workers")
    args = parser.parse_args(argv)
    gs = grid_module(args.grid_repo)
    out = Path(args.out).resolve()
    study = load_study(gs, args.grid_repo, out)
    datasets = {task_key(ds): ds for ds in study["datasets"]}
    jobs = []
    for spec in args.jobs:
        key, family = spec.split("/")
        if key not in datasets or family not in study["families"]:
            raise ValueError(f"{spec} is not a job of this grid")
        if (out / key / family / "completed.json").exists():
            raise ValueError(f"{spec} is already completed")
        jobs.append((datasets[key], family))
    running, failed = {}, set()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        while True:
            queue = []
            for ds, family in jobs:
                if (task_key(ds), family) in failed:
                    continue
                stage, missing = next_trials(gs, study, out, ds, family)
                if stage == "no_winner":
                    failed.add((task_key(ds), family))
                queue += [(ds, family, stage, cid, seed) for cid, seed in missing
                          if (task_key(ds), family, stage, cid, seed) not in running.values()]
            while queue and len(running) < concurrency(args):
                ds, family, stage, cid, seed = queue.pop(0)
                future = pool.submit(fit_trial, str(Path(args.grid_repo).resolve()), study, ds, family, str(out),
                                     stage, cid, seed, stage == "evaluation")
                running[future] = (task_key(ds), family, stage, cid, seed)
                print(time.strftime("%H:%M:%S"), "start", *running[future], flush=True)
            if not running:
                break
            done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            for future in done:
                trial = running.pop(future)
                try:
                    status = future.result()
                except Exception as error:  # a worker crash, not a trial failure (run_trial records those)
                    status = f"crashed: {error!r}"
                print(time.strftime("%H:%M:%S"), status, *trial, flush=True)
                if status != "completed":
                    failed.add(trial[:2])
    for key, family in sorted(failed):
        print(f"{key}/{family}: stopped after a failed trial; the grid's own run handles it", flush=True)
    print("prefill finished; rerun the grid command to write the job summaries", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
