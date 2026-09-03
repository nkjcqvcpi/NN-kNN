"""Harvest the p17 capacity CPU replication into a table + manifest.

(capacity, seed) come from the nested config.config of each run. Wall clock is
derived from the run itself -- the run-dir name carries the launch timestamp and
summary.json is written at completion. The campaign_timing.csv ledger is NOT
trusted: PowerShell's {N1} format emits thousands separators ("1,234.5"), and
that comma corrupts the CSV column alignment.
"""
import datetime, io, json, os, re, statistics, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "rl_capacity_cpu_g234")
DIRSTAMP = re.compile(r"_(\d{8})_(\d{6})_(\d{6})")

def run_seconds(dirpath, name):
    m = DIRSTAMP.search(name)
    sump = os.path.join(dirpath, "summary.json")
    if not m or not os.path.exists(sump):
        return None
    try:
        start = datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        start += datetime.timedelta(microseconds=int(m.group(3)))
    except ValueError:
        return None
    # The run-dir stamp is UTC (manifest.json calls it created_at_utc) but
    # getmtime is local, so compare in UTC or the delta comes out negative.
    end = datetime.datetime.fromtimestamp(
        os.path.getmtime(sump), datetime.timezone.utc).replace(tzinfo=None)
    secs = (end - start).total_seconds()
    return secs if secs > 0 else None


# CUDA reference arms, quoted from reports/capacity_ablation.md (the r760
# campaign). These are NOT recomputed by globbing results/rl: that directory
# holds 63 runs at cap 500 alone -- smoke tests, MLP critics, other phases, and
# duplicate runs at the same seed -- so a glob-and-average gives cap 100 =
# 421.11 +/- 190.39 instead of the real arm's 498.83 +/- 2.62. The canonical
# per-seed values live in the report and reports/phase4_manifest.json.
CUDA_REF = {
    100:  dict(n=5, mean=498.83, sd=2.62,  per=[500.00, 500.00, 494.15, 500.00, 500.00]),
    500:  dict(n=5, mean=497.98, sd=4.52,  per=[500.00, 500.00, 500.00, 500.00, 489.90]),
    2000: dict(n=3, mean=471.53, sd=14.35, per=[455.90, 484.10, 474.60]),
}

runs = []
for name in sorted(os.listdir(OUT)):
    d = os.path.join(OUT, name)
    cfg_p, sum_p = os.path.join(d, "config.json"), os.path.join(d, "summary.json")
    if not (os.path.isdir(d) and os.path.exists(cfg_p) and os.path.exists(sum_p)):
        continue
    cfg = json.load(io.open(cfg_p, encoding="utf-8"))
    inner = cfg.get("config", cfg)
    s = json.load(io.open(sum_p, encoding="utf-8"))
    cap, seed = inner.get("case_capacity"), inner.get("seed")
    if cap is None or seed is None:
        print("  WARN: no capacity/seed in %s" % name); continue
    runs.append(dict(
        run_dir="results/rl_capacity_cpu_g234/" + name,
        capacity=int(cap), seed=int(seed), device=cfg.get("device"),
        final_eval=s["final_eval"]["mean_return"],
        last_eval=s["last_eval"]["mean_return"],
        steps=s["actual_timesteps"],
        best_step=s["training_efficiency"]["best_model_step"],
        stop=s["early_stopping"]["stopping_reason"],
        seconds=run_seconds(d, name),
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

man = os.path.join(ROOT, "reports", "p17_capacity_cpu_manifest.json")
json.dump({"campaign": "p17", "host": "g234", "device": "cpu",
           "protocol": {"profile": "fast", "budget": 150000, "early_stopping": True,
                        "omp_num_threads": 2,
                        "concurrency": "4 jobs (one per capacity) per seed-batch"},
           "summary": rows,
           "runs": sorted(runs, key=lambda r: (r["capacity"], r["seed"]))},
          io.open(man, "w", encoding="utf-8"), indent=2, sort_keys=True)
print("\nwrote reports/p17_capacity_cpu_manifest.json  (%d runs)" % len(runs))
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
print("\nNote: the CUDA arms ran with early stopping on a contended H100; the CPU")
print("arms ran on an idle 16-core box, 4 jobs concurrent (one per capacity).")
print("Returns are comparable; the steps/s columns are NOT comparable across hosts.")

