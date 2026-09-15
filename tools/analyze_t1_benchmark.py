import csv
from pathlib import Path
import numpy as np

def main():
    p = Path("results/t1_benchmark/benchmark_summary.csv")
    if not p.exists():
        print("Summary CSV not found.")
        return

    rows = []
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by (dataset, policy, adapter, mcb)
    groups = {}
    for r in rows:
        key = (r["dataset"], r["policy"], r["adapter"], r["mcb"] == "True")
        groups.setdefault(key, []).append(r)

    print("| Dataset | Retention Policy | Adapter Mode | MCB | Pre-Acc (mean+/-std) | Post-Acc (mean+/-std) | Net Flips |")
    print("|---|---|---|---|---|---|---|")

    for key, items in sorted(groups.items()):
        ds, pol, ad, mcb = key
        pre_accs = [float(x["pre_accuracy"]) for x in items]
        post_accs = [float(x["post_accuracy"]) for x in items]
        net_flips = [int(x["net_flip_benefit"]) for x in items]

        pre_m, pre_s = np.mean(pre_accs), np.std(pre_accs)
        post_m, post_s = np.mean(post_accs), np.std(post_accs)
        flips_m = np.mean(net_flips)

        mcb_str = "ON" if mcb else "OFF"
        print(f"| {ds} | {pol} | {ad} | {mcb_str} | {pre_m:.3f} +/- {pre_s:.3f} | {post_m:.3f} +/- {post_s:.3f} | {flips_m:+.1f} |")

if __name__ == "__main__":
    main()
