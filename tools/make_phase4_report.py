"""Build the Phase 4 fixed-budget comparison table and learning-curve figures.

Usage:
    .venv/bin/python tools/make_phase4_report.py --manifest reports/phase4_manifest.json

The manifest maps method -> {seed: run_dir}. Emits a markdown table on stdout
and writes PNGs under reports/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["dqn", "nec", "nnknn_hybrid", "mlp_ac"]
METHOD_LABEL = {
    "dqn": "DQN",
    "nec": "NEC",
    "nnknn_hybrid": "NN-kNN-RL (hybrid)",
    "mlp_ac": "MLP actor-critic",
}
# Validated categorical palette (dataviz reference instance, light mode).
METHOD_COLOR = {
    "dqn": "#2a78d6",
    "nec": "#eb6834",
    "nnknn_hybrid": "#1baf7a",
    "mlp_ac": "#eda100",
}
SUCCESS_THRESHOLD = 475.0  # overridden by --success-threshold


def load_runs(manifest: dict) -> dict:
    data = {}
    for method, seeds in manifest.items():
        data[method] = {}
        for seed, run_dir in sorted(seeds.items(), key=lambda kv: int(kv[0])):
            d = Path(run_dir)
            summary = json.loads((d / "summary.json").read_text())
            evals = pd.read_csv(d / "eval_metrics.csv")
            data[method][int(seed)] = {"summary": summary, "evals": evals, "dir": str(d)}
    return data


def steps_to_threshold(evals: pd.DataFrame, threshold: float) -> int | None:
    hit = evals[evals["mean_return"] >= threshold]
    return int(hit["global_step"].iloc[0]) if len(hit) else None


def fmt_pm(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}"


def build_table(data: dict, threshold: float) -> str:
    lines = [
        f"| method | final_eval (mean ± std) | last_eval (mean ± std) | best_model_step (mean ± std) | steps-to-{threshold:g} per seed (eval grid) |",
        "|---|---|---|---|---|",
    ]
    for method in METHOD_ORDER:
        runs = data[method]
        finals = [r["summary"]["final_eval"]["mean_return"] for r in runs.values()]
        lasts = [r["summary"]["last_eval"]["mean_return"] for r in runs.values()]
        bests = [r["summary"]["training_efficiency"]["best_model_step"] for r in runs.values()]
        thresh = [steps_to_threshold(r["evals"], threshold) for r in runs.values()]
        thresh_txt = ", ".join("—" if t is None else f"{t // 1000}k" for t in thresh)
        lines.append(
            f"| {METHOD_LABEL[method]} | {fmt_pm(finals)} | {fmt_pm(lasts)} | "
            f"{np.mean(bests):,.0f} ± {np.std(bests, ddof=1):,.0f} | {thresh_txt} |"
        )
    return "\n".join(lines)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e8e7e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5f5e56", labelsize=9)


def plot_per_method(data: dict, outdir: Path, *, task: str, threshold: float,
                    ylim: tuple[float, float], prefix: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHOD_ORDER):
        color = METHOD_COLOR[method]
        for seed, run in sorted(data[method].items()):
            ev = run["evals"]
            ax.plot(ev["global_step"] / 1000, ev["mean_return"], color=color,
                    linewidth=1.6, alpha=0.55)
        ax.axhline(threshold, color="#5f5e56", linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(0.02, 0.955, METHOD_LABEL[method], transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#1a1a19", va="top")
        ax.text(0.02, 0.87, "5 seeds, fixed 150k budget", transform=ax.transAxes,
                fontsize=8.5, color="#5f5e56", va="top")
        style_axis(ax)
        ax.set_ylim(*ylim)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (thousands)", fontsize=9, color="#5f5e56")
    for ax in axes[:, 0]:
        ax.set_ylabel("eval mean return (20 eps)", fontsize=9, color="#5f5e56")
    fig.suptitle(f"{task} fixed-budget evaluation curves — dashed line marks {threshold:g} success threshold",
                 fontsize=11, color="#1a1a19")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = outdir / f"{prefix}_learning_curves_per_method.png"
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return out


def plot_means(data: dict, outdir: Path, *, task: str, threshold: float,
               ylim: tuple[float, float], prefix: str) -> Path:
    # Eval steps differ per run (episode-boundary evals), so resample every
    # seed curve onto one grid before aggregating across seeds.
    grid = np.arange(2_500, 150_001, 2_500)
    label_gap = (ylim[1] - ylim[0]) * 0.046
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    label_pos = []
    for method in METHOD_ORDER:
        color = METHOD_COLOR[method]
        curves = []
        for run in data[method].values():
            ev = run["evals"].drop_duplicates(subset="global_step", keep="last")
            ev = ev.sort_values("global_step")
            curves.append(np.interp(grid, ev["global_step"], ev["mean_return"]))
        aligned = np.vstack(curves)
        steps = grid / 1000
        mean = aligned.mean(axis=0)
        ax.fill_between(steps, aligned.min(axis=0), aligned.max(axis=0),
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2.0)
        label_pos.append([method, float(mean[-1])])
    # Stagger direct labels so line-end names never collide.
    label_pos.sort(key=lambda kv: kv[1])
    for i in range(1, len(label_pos)):
        label_pos[i][1] = max(label_pos[i][1], label_pos[i - 1][1] + label_gap)
    for method, y in label_pos:
        ax.annotate(METHOD_LABEL[method], xy=(grid[-1] / 1000, y),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#1a1a19", va="center")
    ax.axhline(threshold, color="#5f5e56", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1, threshold + label_gap * 0.35, f"{threshold:g} success threshold",
            fontsize=8.5, color="#5f5e56")
    style_axis(ax)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.18)  # room for direct labels
    ax.set_ylim(*ylim)
    ax.set_xlabel("environment steps (thousands)", fontsize=9.5, color="#5f5e56")
    ax.set_ylabel("eval mean return (20 episodes)", fontsize=9.5, color="#5f5e56")
    ax.set_title(f"{task} — seed-mean evaluation return at matched 150k budgets (band = min–max over 5 seeds)",
                 fontsize=11, color="#1a1a19")
    fig.tight_layout()
    out = outdir / f"{prefix}_learning_curves_mean.png"
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/phase4_manifest.json")
    parser.add_argument("--outdir", default="reports/figures")
    parser.add_argument("--task", default="CartPole-v1", help="Task label used in figure titles.")
    parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD)
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.0, 525.0))
    parser.add_argument("--prefix", default="phase4", help="Figure filename prefix.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_runs(manifest)
    print(build_table(data, args.success_threshold))
    print()
    kwargs = dict(task=args.task, threshold=args.success_threshold,
                  ylim=tuple(args.ylim), prefix=args.prefix)
    print(f"figure: {plot_per_method(data, outdir, **kwargs)}")
    print(f"figure: {plot_means(data, outdir, **kwargs)}")


if __name__ == "__main__":
    main()
