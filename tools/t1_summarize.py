"""Summarize a T1 experiment from its raw ``runs__*.jsonl`` shards (no narrative, numbers only).

    python tools/t1_summarize.py results/t1/<experiment> --metric test_accuracy_pre \
        --reference random_K0.5 [--metric test_accuracy_post ...]

Writes ``summary.csv`` (mean, sd, n, 95% t-interval per dataset x condition x metric)
and, with ``--reference``, ``paired_vs_<reference>.csv`` (per-seed paired differences,
mean difference, 95% t-interval, paired t-test p, number of seeds where the condition
beats the reference). Conclusions are left to the reader; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def t_interval(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se)
    return (m - h, m + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--metric", action="append", required=True)
    ap.add_argument("--reference", default=None, help="condition label used as the paired reference")
    ap.add_argument("--higher-is-better", action="store_true", default=True)
    args = ap.parse_args()
    exp = Path(args.exp_dir)
    rows = [json.loads(l) for f in sorted(exp.glob("runs*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
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
