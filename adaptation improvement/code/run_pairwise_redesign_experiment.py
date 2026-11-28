"""Launch disjoint, recorded CPU jobs for the pairwise adapter redesign."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
from run_local_comparison import LOCAL_TASKS


def main(*, driver="tools/t1_pairwise_redesign.py"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=LOCAL_TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    executable = project / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
    uv = str(executable) if executable.is_file() else shutil.which("uv")
    if not uv:
        parser.error("uv unavailable")
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(workspace / ".venv"),
        UV_CACHE_DIR=str(project / ".tools" / "cache"),
        UV_PYTHON_INSTALL_DIR=str(project / ".tools" / "python"),
        NNKNN_RUNS=str(workspace / "runs"),
        NNKNN_DATASETS=str(workspace.parent / "NN-kNN" / "datasets"),
        NNKNN_DEVICE="cpu", NNKNN_DETERMINISTIC="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    stage = project / "code" / "neural-cbr"
    command = [uv, "run", "--project", str(workspace), "--no-sync", "python", "-u",
        driver, "--study", args.study]
    if args.mode == "summarize":
        return subprocess.call(command + ["--mode", "summarize", "--tasks", *args.tasks,
            "--seeds", *map(str, args.seeds)], cwd=stage, env=env)
    logs = project / "logs" / args.study
    logs.mkdir(parents=True, exist_ok=True)
    tasks = sorted(args.tasks, key=lambda t: (t != "regression__car", t != "classification__digits", t))
    jobs = [(task, seed) for task in tasks for seed in args.seeds]

    def job(task, seed):
        log = logs / f"{task}_s{seed}.log"
        with log.open("a", encoding="utf-8") as output:
            code = subprocess.call(command + ["--tasks", task, "--seeds", str(seed)],
                cwd=stage, env=env, stdout=output, stderr=subprocess.STDOUT)
        return code, log.name

    print(f"[start] {len(jobs)} jobs, {args.workers} workers; {driver}", flush=True)
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
    print("All requested runs complete. Generate summaries separately with --mode summarize.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
