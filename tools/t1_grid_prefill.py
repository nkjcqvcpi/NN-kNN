"""Run the pending trials of t1_grid_search.py jobs concurrently; the grid's own run then reuses them.

t1_grid_search.py gives each (dataset, family) job one worker and fits its trials one after
another, so cores sit idle when a few slow jobs remain. This tool fits the same trials with the
grid's own ``run_trial`` (same run units and config), several at a time, stage by stage:
every screen trial; then the refit of the finalists that the grid's ``rank_candidates`` picks from
those trials; then the evaluation of the winner it picks from the refits. Afterwards rerun the
grid's full command unchanged: it reads every complete trial back and writes completed.json itself.

All grid logic is loaded from ``--grid-repo`` (default: this checkout). Its source fingerprint may differ from the
grid's study.json (owner's order 2026-11-23: same configuration, not same code); each trial records the source that
fitted it, and trials recorded under another source are reused as they are. The grid must record its trials as run units
(``--study S``, so its study.json holds ``"study": S``, and the checkout's ``run_trial`` takes ``study``); the
prefill fits the trials into the same units and reads the complete units' ``trial`` records. Grids that wrote
trial files (receipts) are refused. Stop the grid's own run on these jobs first.

    python tools/t1_grid_prefill.py --grid-repo CHECKOUT --out GRID_OUT --jobs TASK_KEY/FAMILY.. \
        [--workers N] [--workers-file PATH]

Jobs are served in the order given (put the longest first). ``--workers-file`` holds an integer
that is re-read before each submission, so the concurrency can be raised or lowered while running
(0 pauses). Like dataset_family, the tool writes shortlist.json before the refits and selection.json
before any evaluation (test-stream) trial; the grid rewrites both unchanged. When a worker process
dies the pool kills the other workers; their trials run again in a new pool.
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
from concurrent.futures.process import BrokenProcessPool

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
    """The grid's frozen study, refused unless it is consistent and was started with this determinism setting. The
    source fingerprint of ``repo`` is not compared: it is provenance, not identity."""
    import inspect
    if "source" in inspect.signature(gs.run_trial).parameters:  # the grid code before 2026-11-23: other arguments
        raise ValueError(f"{repo}: its run_trial still takes the source fingerprint; use a checkout whose trial "
                         "config holds no source (identity is the configuration), e.g. this one (omit --grid-repo)")
    frozen = json.loads((Path(out) / "study.json").read_text())
    study = dict(frozen["configuration"])
    if gs.config_id(study) != frozen["configuration_id"]:
        raise ValueError("study.json configuration does not match its configuration_id")
    if frozen.get("deterministic_algorithms", False) != deterministic_requested():
        raise ValueError("NNKNN_DETERMINISTIC differs from the grid's study.json")
    study["source_sha256"] = frozen["source_sha256"]
    return study


def recorded_study(out):
    """The run study a grid records its trials into (``"study"`` in its study.json); a grid without one wrote trial
    files, which T1 no longer writes or reads, and is refused."""
    study = json.loads((Path(out) / "study.json").read_text()).get("study")
    if not study:
        raise ValueError(f"{out}: the grid records no run study (started without --study); it cannot be prefilled")
    return study


def fit_trial(repo, study, ds, family, out, stage, cid, seed, test, run_study):
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
        import inspect
        if "study" not in inspect.signature(getattr(gs, "run_trial", lambda: None)).parameters:
            raise ValueError("the grid records run units but its pinned checkout's run_trial cannot")
        row = gs.run_trial(job, family, cid, grid[cid], stage, gs.split_loader(study, ds), seed, epochs, patience,
                           test, study=run_study)
    return row["status"]


def receipt(job, stage, cid, seed, run_study, config):
    """The ``trial`` record of the trial's run unit when it is complete with ``config`` (the grid's
    ``trial_config``), else None; a unit complete with another config raises ConfigMismatchError."""
    from model.common.tracking import load_run, resolve_unit
    from model.t1.records import key, unordered
    info = resolve_unit(key(run_study, Path(job).parent.name, Path(job).name, stage, cid, f"s{seed}"), config=config)
    if info is None:
        return None
    return unordered(load_run(info.path, arrays=False).record("trial"))


def write_frozen(gs, path, obj):
    """Write a selection record as the grid would, before the stage that depends on it.

    The grid rewrites the same dict when it is rerun. An existing file that holds anything else is refused; its
    ``source_sha256`` (provenance) is not compared.
    """
    if path.exists():
        strip = lambda d: {k: v for k, v in d.items() if k != "source_sha256"}  # noqa: E731
        if strip(json.loads(path.read_text())) != strip(json.loads(json.dumps(obj))):
            raise ValueError(f"{path} disagrees with the selection recomputed from the receipts")
        return
    gs.write_json(path, obj)


def next_trials(gs, study, out, ds, family, running=(), *, run_study):
    """(stage, [(cid, seed), ..]) still missing at the job's first unfinished stage; ("done", []) at the end.

    ``running`` holds the job's in-flight (stage, cid, seed): they count as missing and their receipts, which may be
    mid-write, are not read. Finalists and the winner come from the grid's rank_candidates over the receipts, and
    shortlist.json, selection.json (before any test-stream trial) or failed.json are written as dataset_family
    writes them.
    """
    job = Path(out) / task_key(ds) / family
    seeds = study["development_seeds"]
    grid = {gs.config_id(p): p for p in gs.expand_grid(study["grids"][family])}

    data = gs.split_loader(study, ds)  # lazy: only its data definition is used

    def config(stage, cid, seed):
        epochs, patience = ((study["screen_epochs"], study["screen_patience"]) if stage == "screen"
                            else (study["refit_epochs"], study["refit_patience"]))
        return gs.trial_config(job, family, cid, grid[cid], stage, data, seed, epochs, patience,
                               stage == "evaluation", torch_threads=study["torch_threads"])

    def receipts(stage, trials):
        return [None if (stage, *t) in running else receipt(job, stage, *t, run_study=run_study, config=config(stage, *t))
                for t in trials]

    screen = [(cid, s) for cid in grid for s in seeds]
    rows = receipts("screen", screen)
    if None in rows:
        return "screen", [t for t, r in zip(screen, rows) if r is None]
    finalists = gs.rank_candidates(rows, grid, seeds)[:study["finalists"]]
    write_frozen(gs, job / "shortlist.json", dict(candidate_ids=finalists, selection_stream="development_validation_only"))
    refit = [(cid, s) for cid in finalists for s in seeds]
    rows = receipts("refit", refit)
    if None in rows:
        return "refit", [t for t, r in zip(refit, rows) if r is None]
    ranked = gs.rank_candidates(rows, finalists, seeds)
    if not ranked:
        write_frozen(gs, job / "failed.json", dict(reason="No candidate completed on every development seed"))
        return "no_winner", []
    winner = ranked[0]
    # The winner is frozen on disk before any test-stream trial, as in dataset_family.
    write_frozen(gs, job / "selection.json", dict(candidate_id=winner, parameters=grid[winner],
        development_seeds=seeds, test_stream_used=False, source_sha256=study["source_sha256"]))
    evaluation = [(winner, s) for s in study["evaluation_seeds"]]
    missing = [t for t, r in zip(evaluation, receipts("evaluation", evaluation)) if r is None]
    return ("evaluation", missing) if missing else ("done", [])


class Concurrency:
    """--workers, or the integer in --workers-file re-read on every call; unreadable content keeps the last value."""

    def __init__(self, args):
        self.path, self.value, self.ignored = args.workers_file, args.workers, None

    def __call__(self):
        if self.path:
            try:
                self.value = max(0, int(Path(self.path).read_text(encoding="utf-8-sig").strip()))
            except (OSError, ValueError) as error:
                if repr(error) != self.ignored:
                    self.ignored = repr(error)
                    log(f"workers file ignored ({error}); keeping {self.value}")
        return self.value


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


CUT_OFF = "cut off by a dead worker process"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-repo", default=str(Path(__file__).resolve().parents[1]),
                        help="checkout whose grid code fits the trials (default: this one); any code state")
    parser.add_argument("--out", required=True, help="the grid's output directory")
    parser.add_argument("--jobs", nargs="+", required=True, help="TASK_KEY/FAMILY, longest first")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--workers-file", help="integer re-read before each submission; 0 pauses")
    parser.add_argument("--max-workers", type=int, default=16, help="process pool size; the active limit is --workers")
    parser.add_argument("--max-pool-restarts", type=int, default=3, help="new pools allowed after worker processes die")
    args = parser.parse_args(argv)
    gs = grid_module(args.grid_repo)
    repo = str(Path(args.grid_repo).resolve())
    out = Path(args.out).resolve()
    study = load_study(gs, args.grid_repo, out)
    run_study = recorded_study(out)
    datasets = {task_key(ds): ds for ds in study["datasets"]}
    jobs = []
    for spec in dict.fromkeys(args.jobs):
        key, family = spec.split("/")
        if key not in datasets or family not in study["families"]:
            raise ValueError(f"{spec} is not a job of this grid")
        if (out / key / family / "completed.json").exists():
            raise ValueError(f"{spec} is already completed")
        jobs.append((datasets[key], family))
    concurrency, running, failed, state, restarts = Concurrency(args), {}, set(), {}, 0
    pool = ProcessPoolExecutor(max_workers=args.max_workers)
    try:
        while True:
            queue = []
            for ds, family in jobs:
                key = task_key(ds)
                if (key, family) in failed:
                    continue
                mine = {t[2:] for t in running.values() if t[:2] == (key, family)}
                stage, missing = next_trials(gs, study, out, ds, family, mine, run_study=run_study)
                state[key, family] = stage
                if stage == "no_winner":
                    failed.add((key, family))
                queue += [(ds, family, stage, cid, seed) for cid, seed in missing if (stage, cid, seed) not in mine]
            broken = None
            while queue and len(running) < concurrency():
                ds, family, stage, cid, seed = queue.pop(0)
                try:
                    future = pool.submit(fit_trial, repo, study, ds, family, str(out), stage, cid, seed,
                                         stage == "evaluation", run_study)
                except BrokenProcessPool as error:
                    broken = error
                    break
                running[future] = (task_key(ds), family, stage, cid, seed)
                log("start", *running[future])
            if not running and not broken:
                if not queue:
                    break
                time.sleep(60)  # paused: the workers file holds 0
                continue
            if not broken:
                done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
                for future in done:
                    trial = running.pop(future)
                    try:
                        status = future.result()
                    except BrokenProcessPool as error:  # a worker process died and the pool killed the others
                        broken, status = error, CUT_OFF
                    except Exception as error:  # an error outside the trial, which records its own failures
                        status = f"crashed: {error!r}"
                    log(status, *trial)
                    if status not in ("completed", CUT_OFF):
                        failed.add(trial[:2])
            if broken:
                for trial in running.values():
                    log(CUT_OFF, *trial)
                running.clear()
                pool.shutdown(wait=True, cancel_futures=True)
                restarts += 1
                if restarts > args.max_pool_restarts:
                    raise RuntimeError(f"worker processes died {restarts} times; last error {broken!r}")
                log(f"worker process died ({broken!r}); new pool {restarts}/{args.max_pool_restarts}; "
                    "its unfinished trials run again")
                pool = ProcessPoolExecutor(max_workers=args.max_workers)
    except BaseException as error:
        if running:
            log(f"stopping ({error!r}); waiting for {len(running)} running trial(s) to write their receipts")
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for (key, family), stage in state.items():
        note = " (stopped after a failed trial; the grid's own run retries it)" if (key, family) in failed else ""
        log(f"{key}/{family}: {stage}{note}")
    log("once every prefill instance on this --out has exited, rerun the grid's full command: it reuses the "
        "receipts and writes completed.json")
    return 0 if all(stage == "done" for stage in state.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
