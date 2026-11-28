"""Launch the recorded local query-aware adapter and conventional baselines."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from run_local_comparison import LOCAL_TASKS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",default="aq1")
    parser.add_argument("--mode",choices=("run","data","query","baselines","summarize"),default="run")
    parser.add_argument("--tasks",nargs="+",default=LOCAL_TASKS)
    parser.add_argument("--seeds",nargs="+",type=int,default=[40,41,42])
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    workspace=project.parent
    path=project/".tools"/("uv.exe" if os.name=="nt" else "uv")
    uv=str(path) if path.is_file() else shutil.which("uv")
    if not uv: parser.error("uv unavailable")
    env=dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(workspace/".venv"),UV_CACHE_DIR=str(project/".tools"/"cache"),
        UV_PYTHON_INSTALL_DIR=str(project/".tools"/"python"),NNKNN_RUNS=str(workspace/"runs"),
        NNKNN_DATASETS=str(workspace.parent/"NN-kNN"/"datasets"),NNKNN_DEVICE="cpu",NNKNN_DETERMINISTIC="1",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg")
    command=[uv,"run","--project",str(workspace),"--no-sync","python","-u","tools/t1_query_aware.py",
             "--study",args.study,"--tasks",*args.tasks,"--seeds",*[str(s) for s in args.seeds]]
    stage=project/"code"/"neural-cbr"
    result=subprocess.call(command+["--mode",args.mode],cwd=stage,env=env)
    if result==0 and args.mode=="run": result=subprocess.call(command+["--mode","summarize"],cwd=stage,env=env)
    return result


if __name__=="__main__": raise SystemExit(main())
