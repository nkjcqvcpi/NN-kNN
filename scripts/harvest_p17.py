"""Harvest the p17 capacity CPU replication into a table + manifest.

Reads only run units of NN-KNN/runs (model/t2/runs.py): every complete unit under
RUNS_FOLDER (searched recursively, e.g. runs/t2/<study>/nnknn_rl/cartpole) is one
run; its config.json and summary.json come from the unit's records and
(capacity, seed) from the nested config.config. The legacy run directories under
results/ are not read any more (their manifest, made by the earlier version of
this script, stays where it is and is not regenerated).

"seconds" is the recorded training time (x/sys/train_seconds: the training call
only, summed over the unit's sessions; an interrupted session counts up to its
last checkpoint), not the launch-to-summary.json wall clock the legacy harvest
derived from run-directory names. Do not compare the two in one throughput table.

Usage: python scripts/harvest_p17.py RUNS_FOLDER [MANIFEST]; the manifest
(generated, not in git) defaults to results/t2/p17_capacity_cpu_manifest.json in
this worktree.
"""
import io, json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository (this file is in scripts/)
sys.path.insert(0, ROOT)

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/harvest_p17.py RUNS_FOLDER [MANIFEST]  (RUNS_FOLDER: a folder of run units)")
OUT = sys.argv[1]
MAN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "t2", "p17_capacity_cpu_manifest.json")


# CUDA reference arms, quoted from reports/capacity_ablation.md (the r760
# campaign). These are NOT recomputed by globbing results/r760: that directory
# holds 63 runs at cap 500 alone -- smoke tests, MLP critics, other phases, and
# duplicate runs at the same seed -- so a glob-and-average gives cap 100 =
# 421.11 +/- 190.39 instead of the real arm's 498.83 +/- 2.62. The canonical
# per-seed values live in the report and reports/phase4_manifest.json.
CUDA_REF = {
    100:  dict(n=5, mean=498.83, sd=2.62,  per=[500.00, 500.00, 494.15, 500.00, 500.00]),
    500:  dict(n=5, mean=497.98, sd=4.52,  per=[500.00, 500.00, 500.00, 500.00, 489.90]),
    # n=5 as of 2026-09-05: r760's p14 seeds 3-4 finally landed (498.20, 487.80),
    # both higher than any of the first three. The published n=3 mean of
    # 471.53 was the worst three of five.
    2000: dict(n=5, mean=480.12, sd=15.96, per=[455.90, 484.10, 474.60, 498.20, 487.80]),
}

from model.common.tracking import iter_runs
from model.t2.runs import RunFiles

runs = []
for info in sorted(iter_runs(OUT), key=lambda i: str(i.path)):
    d = str(info.path)
    try:
        unit = RunFiles(d)
        cfg, s = unit.json("config.json"), unit.json("summary.json")
    except FileNotFoundError as exc:  # e.g. an evaluation unit of the same study
        print("  skip: %s" % exc); continue
    inner = cfg.get("config", cfg)
    cap, seed = inner.get("case_capacity"), inner.get("seed")
    if cap is None or seed is None:
        print("  WARN: no capacity/seed in %s" % d); continue
    rel = os.path.relpath(d, ROOT)  # as the manifest names runs: relative to the repository when inside it
    runs.append(dict(
        run_dir=d if rel.startswith("..") else rel.replace(os.sep, "/"),
        capacity=int(cap), seed=int(seed), device=cfg.get("device"),
        final_eval=s["final_eval"]["mean_return"],
        last_eval=s["last_eval"]["mean_return"],
        steps=s["actual_timesteps"],
        best_step=s["training_efficiency"]["best_model_step"],
        stop=s["early_stopping"]["stopping_reason"],
        seconds=unit.train_seconds(),
    ))

if not runs:
    sys.exit("no completed runs yet")

caps = sorted({r["capacity"] for r in runs})
print("\n## p17 -- capacity ablation, CPU replication on g234 (all runs device=cpu)\n")
print("| capacity | n | final_eval mean +/- sd | per-seed | target-score stops | median steps/s | median wall s |")
print("|---|---|---|---|---|---|---|")
rows = {}
for c in caps:
    rs = sorted([r for r in runs if r["capacity"] == c], key=lambda r: r["seed"])
    vals = [r["final_eval"] for r in rs]
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    stops = sum(1 for r in rs if r["stop"] == "target_score")
    sps = [r["steps"] / r["seconds"] for r in rs if r.get("seconds")]
    secs = [r["seconds"] for r in rs if r.get("seconds")]
    med = statistics.median(sps) if sps else float("nan")
    med_secs = statistics.median(secs) if secs else float("nan")
    per = ", ".join("s%d=%.2f" % (r["seed"], r["final_eval"]) for r in rs)
    print("| %d | %d | **%.2f +/- %.2f** | %s | %d/%d | %.2f | %.0f |"
          % (c, len(rs), mean, sd, per, stops, len(rs), med, med_secs))
    rows[c] = dict(n=len(rs), mean=mean, sd=sd, stops=stops,
                   median_steps_per_s=med, median_wall_seconds=med_secs)

os.makedirs(os.path.dirname(os.path.abspath(MAN)), exist_ok=True)
json.dump({"campaign": "p17", "host": "g234", "device": "cpu",
           "protocol": {"profile": "fast", "budget": 150000, "early_stopping": True,
                        "omp_num_threads": 2,
                        "concurrency": "4 jobs (one per capacity) per seed-batch"},
           "summary": rows,
           "runs": sorted(runs, key=lambda r: (r["capacity"], r["seed"]))},
          io.open(MAN, "w", encoding="utf-8"), indent=2, sort_keys=True)
print("\nwrote %s  (%d runs)" % (MAN, len(runs)))
print("\n## CPU replication (g234) vs the CUDA arms (r760, reports/capacity_ablation.md)\n")
print("| capacity | CUDA r760 (n) | CPU g234 (n) | delta |")
print("|---|---|---|---|")
for c in caps:
    got = rows[c]
    ref = CUDA_REF.get(c)
    if ref:
        print("| %d | %.2f +/- %.2f (%d) | %.2f +/- %.2f (%d) | %+.2f |"
              % (c, ref["mean"], ref["sd"], ref["n"], got["mean"], got["sd"], got["n"],
                 got["mean"] - ref["mean"]))
    else:
        print("| %d | -- (no CUDA arm; this is the new knee point) | %.2f +/- %.2f (%d) | -- |"
              % (c, got["mean"], got["sd"], got["n"]))
print("\nNotes:")
print("- The CUDA arms ran on a contended H100. The CPU arms ran on g234, a")
print("  10-core i5-13400F, at up to 12 concurrent jobs with OMP_NUM_THREADS=2.")
print("  Returns are comparable across hosts; steps/s and wall seconds are NOT --")
print("  use results/t2/p17_throughput_controlled.csv (timing_pass.ps1) for that.")
print("- SURVIVORSHIP WARNING for any arm below n=5: runs that hit the target")
print("  score stop early and therefore land in this table FIRST, biasing a")
print("  partial arm upward. Do not read a partial mean as the arm's result.")
incomplete = [c for c in caps if rows[c]["n"] < 5]
if incomplete:
    print("- INCOMPLETE ARMS (n<5), means are provisional and biased high: %s"
          % ", ".join("cap%d n=%d" % (c, rows[c]["n"]) for c in incomplete))

