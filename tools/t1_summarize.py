"""Summarize a T1 experiment from its run units (no narrative, numbers only).

    python tools/t1_summarize.py <NN-KNN/runs/t1/<study>/<experiment>> --metric test_accuracy_pre \
        --reference random_K0.5 [--metric test_accuracy_post ...] [--out DIR]

The input is the run-unit tree that ``tools/t1_run.py --study`` recorded; the ``row`` records of its condition
units are read in the order the shards of the same runs held them. The outputs go to ``--out``, by default
``<worktree>/results/t1/<study>/<experiment>/`` (``records.results_dir``).

Writes ``summary.csv`` (mean, sd, n, 95% t-interval per dataset x condition x metric)
and, with ``--reference``, ``paired_vs_<reference>.csv`` (per-seed paired differences,
mean difference, 95% t-interval, paired t-test p, number of seeds where the condition
beats the reference). Conclusions are left to the reader; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.t1.records import experiment_rows, require_unit_tree, results_dir  # noqa: E402


def t_interval(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se)
    return (m - h, m + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="run-unit tree of one experiment: NN-KNN/runs/t1/<study>/<experiment>")
    ap.add_argument("--metric", action="append", required=True)
    ap.add_argument("--reference", default=None, help="condition label used as the paired reference")
    ap.add_argument("--higher-is-better", action="store_true", default=True)
    ap.add_argument("--out", default=None, help="output directory (default: <worktree>/results/t1/<study>/<experiment>)")
    args = ap.parse_args()
    src = require_unit_tree(args.exp_dir)
    exp = Path(args.out) if args.out else results_dir(src)
    exp.mkdir(parents=True, exist_ok=True)
    rows = experiment_rows(src)
    df = pd.DataFrame(rows)
    # the latest run per (dataset, condition, seed) wins if a config was re-run
    df = df.drop_duplicates(subset=["dataset", "condition", "seed"], keep="last")
    out = []
    for (ds, cond), g in df.groupby(["dataset", "condition"]):
        for m in args.metric:
            if m not in g:
                continue
            x = g[m].dropna().to_numpy(dtype=float)
            lo, hi = t_interval(x)
            out.append({"dataset": ds, "condition": cond, "metric": m, "mean": x.mean() if len(x) else np.nan, "sd": x.std(ddof=1) if len(x) > 1 else np.nan, "n_seeds": len(x), "ci95_lo": lo, "ci95_hi": hi})
    summ = pd.DataFrame(out)
    summ.to_csv(exp / "summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.reference:
        pr = []
        for ds, g in df.groupby("dataset"):
            ref = g[g["condition"] == args.reference].set_index("seed")
            if ref.empty:
                continue
            for cond, gc in g.groupby("condition"):
                if cond == args.reference:
                    continue
                gc = gc.set_index("seed")
                seeds = sorted(set(ref.index) & set(gc.index))
                for m in args.metric:
                    if m not in gc or m not in ref or len(seeds) < 2:
                        continue
                    d = gc.loc[seeds, m].to_numpy(float) - ref.loc[seeds, m].to_numpy(float)
                    lo, hi = t_interval(d)
                    p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else float("nan")
                    pr.append({"dataset": ds, "condition": cond, "reference": args.reference, "metric": m, "n_paired": len(seeds), "mean_diff": d.mean(), "ci95_lo": lo, "ci95_hi": hi, "paired_t_p": p, "seeds_better": int((d > 0).sum()), "seeds_worse": int((d < 0).sum())})
        pdf = pd.DataFrame(pr)
        pdf.to_csv(exp / f"paired_vs_{args.reference}.csv", index=False)
        print()
        print(pdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
