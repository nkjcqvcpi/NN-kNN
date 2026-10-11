"""Report an ``encoder_ablation`` experiment: does the EMA case encoder help?

    python tools/t1_encoder_report.py <experiment dir> --out <dir> [--reference shared --reference raw]

Error is 1 - accuracy (classification) or train-standardized RMSE (regression);
lower is better. Retrieval-only rows use the pre-adaptation output, all other
settings the adapted output. Per setting, every arm is compared with each
reference arm on paired seeds. ``ema_valsel`` picks one EMA momentum per task
and setting by mean validation error (test never selects). Writes
``encoder_summary.csv``, ``encoder_paired.csv``, ``encoder_stability.csv``
and ``encoder_report.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def setting_of(row: dict) -> str:
    name = row["condition"][len(row["arm"]) + 2:]
    if name == "retrieval_only":
        return "retrieval_only"
    if "schedule" not in row:
        return "frozen_reuse"
    return name


def error(row: dict, stream: str) -> float:
    post = "pre" if setting_of(row) == "retrieval_only" else "post"
    if row["task_type"] == "classification":
        return 1.0 - float(row[f"{stream}_accuracy_{post}"])
    return float(row[f"{stream}_rmse_{post}"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", action="append")
    ap.add_argument("--ema-prefix", default="ema")
    args = ap.parse_args()
    refs = args.reference or ["shared", "raw"]
    rows = []
    for shard in sorted(Path(args.exp_dir).glob("runs__*.jsonl")):
        rows += [json.loads(l) for l in shard.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no rows")
    test, val, task_type = {}, {}, {}
    for r in rows:
        key = (r["dataset"], setting_of(r), r["arm"], r["seed"])
        test[key], val[key] = error(r, "test"), error(r, "val")
        task_type[r["dataset"]] = r["task_type"]
    tasks = sorted({k[0] for k in test})
    settings = [s for s in ("retrieval_only", "frozen_reuse", "alternating_rr", "joint_rr") if any(k[1] == s for k in test)]
    settings += sorted({k[1] for k in test} - set(settings))
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    ema_arms = [a for a in arms if a.startswith(args.ema_prefix)]

    # Validation-selected EMA momentum: one arm per (task, setting), chosen without test.
    chosen = {}
    for t in tasks:
        for s in settings:
            scored = [(np.mean([v for (tt, ss, aa, _), v in val.items() if (tt, ss, aa) == (t, s, a)]), a)
                      for a in ema_arms if any(k[:3] == (t, s, a) for k in val)]
            if scored:
                chosen[t, s] = min(scored)[1]
                for (tt, ss, aa, seed), v in list(test.items()):
                    if (tt, ss, aa) == (t, s, chosen[t, s]):
                        test[t, s, "ema_valsel", seed] = v
    all_arms = arms + (["ema_valsel"] if chosen else [])

    summary = []
    for t in tasks:
        for s in settings:
            for a in all_arms:
                v = [x for (tt, ss, aa, _), x in test.items() if (tt, ss, aa) == (t, s, a)]
                if v:
                    summary.append({"task": t, "task_type": task_type[t], "setting": s, "arm": a,
                                    "test_error_mean": float(np.mean(v)), "test_error_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                    "n_seeds": len(v), "valsel_momentum_arm": chosen.get((t, s), "") if a == "ema_valsel" else ""})
    paired = []
    for s in settings:
        for ref in refs:
            for a in all_arms:
                if a == ref:
                    continue
                for t in tasks:
                    seeds = sorted(k[3] for k in test if k[:3] == (t, s, a) and (t, s, ref, k[3]) in test)
                    if not seeds:
                        continue
                    d = np.array([test[t, s, a, x] - test[t, s, ref, x] for x in seeds])
                    p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and np.std(d) > 0 else float("nan")
                    paired.append({"setting": s, "arm": a, "reference": ref, "task": t, "mean_difference": float(d.mean()),
                                   "seeds_better": int((d < 0).sum()), "seeds_worse": int((d > 0).sum()),
                                   "seeds_tied": int((d == 0).sum()), "paired_t_p": p})
    stability = []
    for r in rows:
        if setting_of(r) == "retrieval_only" or "mean_neighborhood_churn" in r:
            stability.append({"task": r["dataset"], "setting": setting_of(r), "arm": r["arm"], "seed": r["seed"],
                              **{k: r[k] for k in ("core_mean_representation_drift", "core_late_representation_drift",
                                                   "core_mean_neighborhood_churn", "core_late_neighborhood_churn",
                                                   "mean_representation_drift", "mean_neighborhood_churn",
                                                   "online_ema_gap", "core_epochs_run", "core_best_epoch") if k in r}})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "encoder_summary.csv", summary)
    write_csv(out / "encoder_paired.csv", paired)
    write_csv(out / "encoder_stability.csv", stability)

    md = ["# Encoder / EMA ablation", "",
          "Test error (lower is better): 1 - accuracy for classification, train-standardized RMSE for regression. "
          f"Mean over seeds. `ema_valsel` chooses the EMA momentum per task and setting by validation error.", ""]
    for s in settings:
        md += [f"## {s}", "", "| task | " + " | ".join(all_arms) + " |", "|---" * (len(all_arms) + 1) + "|"]
        for t in tasks:
            cells = []
            vals = {a: next((x["test_error_mean"] for x in summary if (x["task"], x["setting"], x["arm"]) == (t, s, a)), None)
                    for a in all_arms}
            best = min(v for v in vals.values() if v is not None)
            for a in all_arms:
                v = vals[a]
                cells.append("" if v is None else (f"**{v:.4f}**" if v == best else f"{v:.4f}"))
            md.append(f"| {t} | " + " | ".join(cells) + " |")
        md.append("")
        for ref in refs:
            parts = []
            for a in all_arms:
                ps = [p for p in paired if (p["setting"], p["arm"], p["reference"]) == (s, a, ref)]
                if ps:
                    parts.append(f"{a}: {sum(p['mean_difference'] < 0 for p in ps)} better / "
                                 f"{sum(p['mean_difference'] > 0 for p in ps)} worse / {sum(p['mean_difference'] == 0 for p in ps)} tied")
            md.append(f"- vs `{ref}` (tasks, by mean paired difference): " + "; ".join(parts))
        md.append("")
    md += ["## Stability during core training (mean over seeds)", "",
           "| task | arm | drift (mean) | neighbor churn (mean) | churn (last 10 epochs) | online-EMA gap |", "|---|---|---|---|---|---|"]
    for t in tasks:
        for a in arms:
            ss = [x for x in stability if x["task"] == t and x["arm"] == a and x["setting"] == "retrieval_only"]
            if ss:
                def m(k):
                    v = [x[k] for x in ss if k in x]
                    return f"{np.mean(v):.4f}" if v else ""
                md.append(f"| {t} | {a} | {m('core_mean_representation_drift')} | {m('core_mean_neighborhood_churn')} | "
                          f"{m('core_late_neighborhood_churn')} | {m('online_ema_gap')} |")
    (out / "encoder_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: 40]))


if __name__ == "__main__":
    main()
