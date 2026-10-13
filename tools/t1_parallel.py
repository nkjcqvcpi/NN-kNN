"""Run one T1 config as independent (dataset, seed) subprocesses with receipts.

    python tools/t1_parallel.py configs/t1/<config>.yaml --out <results dir> --logs <log dir> --workers 8

Each job calls ``tools/t1_run.py --only-dataset <name> --seeds <seed>``. A job whose
receipt records exit code 0 is skipped on resume; failed jobs are retried.
``progress.json`` in the log directory tracks counts. Exits nonzero if any job fails.
Datasets that share a name across task types must be split into separate configs,
because ``--only-dataset`` selects by name.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.t1_run import load_config  # noqa: E402


def run_job(config: Path, name: str, seed: int, out: Path, logs: Path, device: str | None = None,
            key: str | None = None) -> dict:
    log = logs / f"{key or name}__s{seed}.log"
    receipt = log.with_suffix(".json")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["exit_code"] == 0:
            return {**previous, "reused": True}
    env = dict(os.environ)
    if device:
        env["NNKNN_DEVICE"] = device  # tools/t1_run.py defaults to cpu otherwise
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, "-u", str(REPO / "tools/t1_run.py"), str(config),
                               "--only-dataset", name, "--seeds", str(seed), "--out", str(out)],
                              cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
    result = {"config": str(config), "dataset": name, "seed": seed, "exit_code": proc.returncode,
              "seconds": time.time() - t0, "log": str(log), "device": device or env.get("NNKNN_DEVICE", "cpu")}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="+", help="one or more configs; each job uses its config's seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--datasets", nargs="*", help="subset of dataset names, in launch order")
    ap.add_argument("--device", help="NNKNN_DEVICE for every job, e.g. cpu or xpu (one device per batch)")
    args = ap.parse_args()
    multi = len(args.config) > 1
    jobs = []  # (config, dataset name, seed, log key)
    for path in args.config:
        config = Path(path).resolve()
        cfg = load_config(config)
        names = [d["name"] for d in cfg["datasets"]]
        if len(set(names)) != len(names):
            raise SystemExit("dataset names repeat across task types; split the config")
        keys = {d["name"]: f"{d['task_type']}__{d['name']}" if multi else d["name"] for d in cfg["datasets"]}
        jobs += [(config, n, int(s), keys[n]) for n in names for s in (args.seeds or cfg["seeds"])]
    if multi and len({(k, s) for _, _, s, k in jobs}) != len(jobs):
        raise SystemExit("two configs schedule the same task and seed")
    if args.datasets:  # also sets the launch order (start expensive tasks first)
        unknown = set(args.datasets) - {n for _, n, _, _ in jobs}
        if unknown:
            raise SystemExit(f"datasets not in config: {sorted(unknown)}")
        order = {n: i for i, n in enumerate(args.datasets)}
        jobs = sorted((j for j in jobs if j[1] in order), key=lambda j: order[j[1]])
    out, logs = Path(args.out).resolve(), Path(args.logs).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    config = args.config[0] if not multi else f"{len(args.config)} configs"
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, c, n, s, out, logs, args.device, k): (n, s) for c, n, s, k in jobs}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            (logs / "progress.json").write_text(json.dumps({
                "config": str(config), "expected_jobs": len(jobs), "finished_jobs": len(results),
                "passed": sum(x["exit_code"] == 0 for x in results),
                "failed": [x for x in results if x["exit_code"] != 0]}, indent=2))
            print(f"[{len(results)}/{len(jobs)}] {r['dataset']} s{r['seed']}: exit {r['exit_code']}"
                  f" ({r.get('seconds', 0):.0f}s{', reused' if r.get('reused') else ''})", flush=True)
    if any(r["exit_code"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
