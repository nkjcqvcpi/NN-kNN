"""Launch the complete local dropout matrix with disjoint recorded CPU jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
from run_local_comparison import LOCAL_TASKS

ADAPTERS = {"original":"o", "query_aware":"q", "query_no_prediction":"n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="ad1")
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=LOCAL_TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40,41,42])
    parser.add_argument("--adapters", nargs="+", choices=tuple(ADAPTERS), default=list(ADAPTERS))
    parser.add_argument("--rates", nargs="+", type=float, choices=(.1,.2), default=[.1,.2])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    path = project / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
    uv = str(path) if path.is_file() else shutil.which("uv")
    if not uv:
        parser.error("uv unavailable")
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(workspace/".venv"), UV_CACHE_DIR=str(project/".tools"/"cache"),
        UV_PYTHON_INSTALL_DIR=str(project/".tools"/"python"), NNKNN_RUNS=str(workspace/"runs"),
        NNKNN_DATASETS=str(workspace.parent/"NN-kNN"/"datasets"), NNKNN_DEVICE="cpu", NNKNN_DETERMINISTIC="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    stage = project/"code"/"neural-cbr"
    command = [uv, "run", "--project", str(workspace), "--no-sync", "python", "-u",
               "tools/t1_adapter_dropout.py", "--study", args.study]
    summary = command+["--mode", "summarize", "--tasks", *args.tasks, "--seeds", *map(str,args.seeds)]
    if args.mode == "summarize":
        return subprocess.call(summary, cwd=stage, env=env)
    logs = project/"logs"/args.study
    logs.mkdir(parents=True, exist_ok=True)
    tasks = sorted(args.tasks, key=lambda t:(t!="regression__car", t!="classification__digits", t))
    jobs = [(task, seed, adapter, rate) for task in tasks for seed in args.seeds
            for adapter in args.adapters for rate in args.rates]
    def job(task, seed, adapter, rate):
        log = logs/f"{task}_s{seed}_{ADAPTERS[adapter]}_p{round(rate*100):02d}.log"
        with log.open("a", encoding="utf-8") as output:
            result = subprocess.call(command+["--tasks",task,"--seeds",str(seed),"--adapters",adapter,"--rates",str(rate)],
                cwd=stage, env=env, stdout=output, stderr=subprocess.STDOUT)
        return result, log.name
    print(f"[start] {len(jobs)} jobs, {args.workers} workers; three schemes per job", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, *item) for item in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            code, name = future.result()
            if code:
                failures.append(name)
            print(f"[progress] {completed}/{len(jobs)} jobs; exit={code}; {name}", flush=True)
    if failures:
        print(f"Failed jobs: {failures}; complete units are reused on retry.", flush=True)
        return 1
    # A filtered adapter/rate invocation intentionally leaves the full matrix incomplete.
    if set(args.adapters) == set(ADAPTERS) and set(args.rates) == {.1,.2}:
        return subprocess.call(summary, cwd=stage, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
