"""Run the local recorded training-scheme comparison with bounded CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
from run_local_comparison import LOCAL_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="as1")
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=LOCAL_TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    parser.add_argument("--workers", type=int, default=6)
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
    env.update(UV_PROJECT_ENVIRONMENT=str(workspace / ".venv"),
        UV_CACHE_DIR=str(project / ".tools" / "cache"),
        UV_PYTHON_INSTALL_DIR=str(project / ".tools" / "python"),
        NNKNN_RUNS=str(workspace / "runs"),
        NNKNN_DATASETS=str(workspace.parent / "NN-kNN" / "datasets"),
        NNKNN_DEVICE="cpu", NNKNN_DETERMINISTIC="1", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    stage = project / "code" / "neural-cbr"
    command = [uv, "run", "--project", str(workspace), "--no-sync", "python", "-u",
               "tools/t1_adapter_training_schemes.py", "--study", args.study]
    summary = command + ["--mode", "summarize", "--tasks", *args.tasks,
                         "--seeds", *map(str, args.seeds)]
    if args.mode == "summarize":
        return subprocess.call(summary, cwd=stage, env=env)
    logs = project / "logs" / args.study
    logs.mkdir(parents=True, exist_ok=True)
    # Start the largest cases early; every job owns distinct run units.
    tasks = sorted(args.tasks, key=lambda t: (t != "regression__car", t != "classification__digits", t))
    def job(task, seed):
        log = logs / f"{task}_s{seed}.log"
        print(f"[start] {task} s{seed}", flush=True)
        with log.open("a", encoding="utf-8") as output:
            result = subprocess.call(command + ["--tasks", task, "--seeds", str(seed)],
                cwd=stage, env=env, stdout=output, stderr=subprocess.STDOUT)
        print(f"[finished] {task} s{seed}: exit={result}; {log}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, task, seed) for task in tasks for seed in args.seeds]
        failures = [code for future in as_completed(futures) if (code := future.result()) != 0]
    if failures:
        print(f"{len(failures)} jobs failed; complete units will be reused on retry.", flush=True)
        return 1
    return subprocess.call(summary, cwd=stage, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
