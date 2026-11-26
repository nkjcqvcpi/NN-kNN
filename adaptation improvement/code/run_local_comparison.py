"""Run or summarize the local comparison using the project's uv environment.

From the repository root:
    python "adaptation improvement/code/run_local_comparison.py" --mode summarize

This bootstrap uses only the standard library; all ML code runs through uv.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


LOCAL_TASKS = [
    "classification__iris", "classification__wine", "classification__breast_cancer",
    "classification__balance", "classification__digits", "classification__zebra",
    "classification__zebra_special", "regression__airfoil", "regression__car",
    "regression__yacht", "regression__energy_efficiency", "regression__student_performance",
    "regression__body_fat", "regression__diabets",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "preflight", "summarize"), default="run")
    parser.add_argument("--study", default="am1")
    parser.add_argument("--tasks", nargs="+", default=LOCAL_TASKS)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    stage = project / "code" / "neural-cbr"
    executable = project / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
    uv = str(executable) if executable.is_file() else shutil.which("uv")
    if not uv:
        parser.error("uv is unavailable; install uv before running this launcher")
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(workspace / ".venv"),
               UV_CACHE_DIR=str(project / ".tools" / "cache"),
               UV_PYTHON_INSTALL_DIR=str(project / ".tools" / "python"),
               NNKNN_DATASETS=str(workspace.parent / "NN-kNN" / "datasets"),
               NNKNN_RUNS=str(workspace / "runs"), NNKNN_DEVICE="cpu", NNKNN_DETERMINISTIC="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    Path(env["NNKNN_RUNS"]).mkdir(exist_ok=True)
    command = [uv, "run", "--project", str(workspace), "--no-sync", "python", "-u",
               "tools/t1_adaptation_improvement.py", "--study", args.study, "--mode", args.mode,
               "--tasks", *args.tasks]
    return subprocess.call(command, cwd=stage, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
