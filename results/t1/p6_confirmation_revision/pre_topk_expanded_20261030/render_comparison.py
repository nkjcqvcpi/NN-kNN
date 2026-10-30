"""Render all paired results with headroom for the highest data markers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
rows = json.loads((out / "runs.json").read_text())
summary = json.loads((out / "summary.json").read_text())
seeds = sorted({r["seed"] for r in rows})
fig, axes = plt.subplots(4, 3, figsize=(13.5, 13), constrained_layout=True)
for ax, s in zip(axes.flat, summary):
    for seed in seeds:
        values = [next(r["test_error"] for r in rows if r["dataset"] == s["dataset"]
                       and r["seed"] == seed and r["masked"] == flag) for flag in [True, False]]
        ax.plot([0, 1], values, color="#b8b8b8", linewidth=1, zorder=1)
        ax.scatter([0, 1], values, c=["#245e8b", "#bb7619"], s=35, zorder=2, clip_on=False)
    means = [s["masked_mean_error"], s["all_case_mean_error"]]
    ax.scatter([0, 1], means, marker="D", s=65, c=["#245e8b", "#bb7619"], edgecolor="black", zorder=3, clip_on=False)
    ax.set_xticks([0, 1], ["Top-5 mask", "All-case softmax"])
    ax.set_xlim(-0.25, 1.25)
    ymax = max(r["test_error"] for r in rows if r["dataset"] == s["dataset"])
    ax.set_ylim(0, max(ymax * 1.12, 0.01))
    ax.set_title(s["dataset"].replace("_", " ").title())
    ax.set_ylabel("Standardized-target RMSE" if s["task_type"] == "regression" else "Classification error rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("Pre-Top-K masking: 3 paired seeds per dataset\nLower error is better; diamonds are means")
fig.savefig(out / "comparison.png", dpi=160)
plt.close(fig)
