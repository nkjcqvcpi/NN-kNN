"""Read saved test predictions to diagnose adaptation direction and magnitude.

No fitting, model evaluation or run mutation. Output is descriptive hindsight,
not a deployable tuning rule. Each adapter is compared to its own selected core.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np

from model.common.tracking import load_unit
from model.t1 import records
from tools.t1_adaptation_improvement import ComparisonRunner, EXPERIMENT, suite

TASKS = ["classification__" + s for s in (
    "iris", "wine", "breast_cancer", "balance", "digits", "zebra", "zebra_special"
)] + ["regression__" + s for s in (
    "airfoil", "car", "yacht", "energy_efficiency", "student_performance", "body_fat", "diabets"
)]
ARMS = ("existing", "magnitude_direction")
SCALES = (0., .25, .5, 1., 2., 4., 8.)
EPS = 1e-8  # absolute tolerance in standardized target / class-score units


def regression_diagnostics(pre, post, target):
    pre, post, target = (np.asarray(v, dtype=float) for v in (pre, post, target))
    residual, correction = target - pre, post - pre
    before, after = residual ** 2, (residual-correction) ** 2
    delta = after-before
    active = abs(correction) > EPS
    needed = abs(residual) > EPS
    ratio = np.divide(correction, residual, out=np.full_like(residual, np.nan), where=needed)
    category = np.full(len(pre), "unchanged", dtype=object)
    category[active & ~needed] = "moved_from_exact"
    category[active & needed & (ratio < 0)] = "wrong_direction"
    category[active & needed & (ratio > 0) & (ratio < 1)] = "toward_target_short"
    category[active & needed & np.isclose(ratio, 1, atol=EPS, rtol=0)] = "target_reached"
    category[active & needed & (ratio > 1+EPS) & (ratio < 2-EPS)] = "overshoot_still_helps"
    category[active & needed & np.isclose(ratio, 2, atol=EPS, rtol=0)] = "overshoot_equal_error"
    category[active & needed & (ratio > 2+EPS)] = "overshoot_harms"
    c2, rc = float(np.sum(correction**2)), float(np.sum(residual*correction))
    alpha = max(0., rc/c2) if c2 > 0 else 0.
    summary = {
        "n": len(pre), "pre_mse": float(before.mean()), "post_mse": float(after.mean()),
        "pre_error": float(np.sqrt(before.mean())), "post_error": float(np.sqrt(after.mean())),
        "mean_abs_correction": float(abs(correction).mean()), "median_abs_correction": float(np.median(abs(correction))),
        "needed_mae": float(abs(residual).mean()),
        "mean_c2": c2/len(pre), "mean_2rc": 2*rc/len(pre),
        "oracle_nonnegative_global_scale": alpha,
        "oracle_scale_mse": float(np.mean((residual-alpha*correction)**2)),
        "oracle_sign_mse": float(np.mean((residual-np.sign(residual)*abs(correction))**2)),
        "wrong_direction_harm_sse": float(np.sum(delta[category == "wrong_direction"])),
        "overshoot_harm_sse": float(np.sum(delta[category == "overshoot_harms"])),
        "exact_harm_sse": float(np.sum(delta[category == "moved_from_exact"])),
        "total_harm_sse": float(np.sum(np.maximum(delta, 0))),
        "total_benefit_sse": float(np.sum(np.maximum(-delta, 0))),
        "improved": int(np.sum(delta < -EPS)), "worsened": int(np.sum(delta > EPS)),
    }
    if not np.isclose(summary["post_mse"]-summary["pre_mse"], summary["mean_c2"]-summary["mean_2rc"], atol=1e-12):
        raise AssertionError("squared-error decomposition does not reconcile")
    counts = Counter(category)
    summary.update({"count_"+k: int(counts[k]) for k in (
        "unchanged", "moved_from_exact", "wrong_direction", "toward_target_short", "target_reached",
        "overshoot_still_helps", "overshoot_equal_error", "overshoot_harms")})
    cases = [{"category": str(category[i]), "required_residual": float(residual[i]),
              "applied_correction": float(correction[i]), "correction_fraction": float(ratio[i]) if needed[i] else None,
              "pre_squared_error": float(before[i]), "post_squared_error": float(after[i]),
              "squared_error_change": float(delta[i])} for i in range(len(pre))]
    curves = [{"scale": a, "error": float(np.sqrt(np.mean((residual-a*correction)**2)))} for a in SCALES]
    return summary, cases, curves


def correct_scale_interval(pre, correction, target):
    """Positive-scale interval where the target wins against every rival.

    Measure-zero ties are handled separately using recorded argmax outcomes.
    A zero-slope tied rival that wins argmax makes the direction unrecoverable.
    """
    lower, upper = 0., math.inf
    for j in range(len(pre)):
        if j == target:
            continue
        gap, slope = pre[target]-pre[j], correction[target]-correction[j]
        if abs(slope) <= 1e-14:
            if gap < 0 or (gap == 0 and j < target):
                return None
        elif slope > 0:
            lower = max(lower, -gap/slope)
        else:
            upper = min(upper, -gap/slope)
    if upper <= lower or upper <= 0:
        return None
    probe = (lower+upper)/2 if math.isfinite(upper) else lower+max(1., abs(lower)*.01)
    if np.argmax(pre+probe*correction) != target:
        return None
    return lower, upper


def classification_diagnostics(pre, post, target):
    pre, post = (np.asarray(v, dtype=float) for v in (pre, post))
    target = np.asarray(target, dtype=int)
    c = post-pre
    pred0, pred1 = pre.argmax(1), post.argmax(1)
    ok0, ok1 = pred0 == target, pred1 == target
    indices = np.arange(len(target))
    other0, other1 = pre.copy(), post.copy()
    other0[indices, target], other1[indices, target] = -np.inf, -np.inf
    margin0, margin1 = pre[indices, target]-other0.max(1), post[indices, target]-other1.max(1)
    centered = c-c.mean(1, keepdims=True)
    cases = []
    for i, y in enumerate(target):
        interval = correct_scale_interval(pre[i], c[i], int(y))
        lo, hi = interval if interval else (None, None)
        if ok0[i]:
            category = "retained_correct" if ok1[i] else "harmful_flip"
        elif ok1[i]:
            category = "fixed"
        elif interval is None:
            category = "no_positive_scale_rescues"
        elif lo >= 1-EPS:
            category = "needs_larger_scale"
        elif hi <= 1+EPS:
            category = "needs_smaller_scale"
        else:
            raise AssertionError("scale interval conflicts with recorded class decision")
        rival = int(other0[i].argmax())
        cases.append({"category": category, "pre_correct": bool(ok0[i]), "post_correct": bool(ok1[i]),
            "pre_margin": float(margin0[i]), "post_margin": float(margin1[i]),
            "margin_change": float(margin1[i]-margin0[i]),
            "slope_vs_initial_rival": float(c[i,y]-c[i,rival]),
            "correction_l2": float(np.linalg.norm(c[i])), "centered_correction_l2": float(np.linalg.norm(centered[i])),
            "min_scale_to_correct": lo, "max_scale_to_correct": hi if hi is not None and math.isfinite(hi) else None,
            "has_correct_scale": interval is not None,
            "rescuable_up_to_2": interval is not None and max(lo, 0.) < min(hi, 2.),
            "rescuable_up_to_4": interval is not None and max(lo, 0.) < min(hi, 4.),
            "rescuable_up_to_10": interval is not None and max(lo, 0.) < min(hi, 10.)})
    counts = Counter(r["category"] for r in cases)
    summary = {"n": len(target), "pre_error": float(np.mean(~ok0)), "post_error": float(np.mean(~ok1)),
        "pre_wrong": int(np.sum(~ok0)), "post_wrong": int(np.sum(~ok1)),
        "margin_improves": int(np.sum(margin1 > margin0+EPS)), "margin_worsens": int(np.sum(margin1 < margin0-EPS)),
        "mean_correction_l2": float(np.linalg.norm(c, axis=1).mean()),
        "mean_centered_correction_l2": float(np.linalg.norm(centered, axis=1).mean()),
        "common_offset_energy_fraction": float(np.sum((c-centered)**2)/np.sum(c**2)) if np.sum(c**2) > 0 else 0.,
        "unfixed_recoverable_up_to_2": int(sum(not r["post_correct"] and not r["pre_correct"] and r["rescuable_up_to_2"] for r in cases)),
        "unfixed_recoverable_up_to_4": int(sum(not r["post_correct"] and not r["pre_correct"] and r["rescuable_up_to_4"] for r in cases)),
        "unfixed_recoverable_up_to_10": int(sum(not r["post_correct"] and not r["pre_correct"] and r["rescuable_up_to_10"] for r in cases)),
    }
    summary.update({"count_"+k: int(counts[k]) for k in ("retained_correct", "harmful_flip", "fixed",
        "no_positive_scale_rescues", "needs_larger_scale", "needs_smaller_scale")})
    if summary["pre_wrong"]-summary["count_fixed"]+summary["count_harmful_flip"] != summary["post_wrong"]:
        raise AssertionError("classification flip counts do not reconcile")
    curves = [{"scale": a, "error": float(np.mean(np.argmax(pre+a*c, axis=1) != target))} for a in SCALES]
    return summary, cases, curves


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(study, tasks=TASKS, seeds=(40, 41, 42)):
    configs = suite(tasks=tasks)
    units, cases, curves, histories, provenance = [], [], [], [], []
    for task, cfg in configs.items():
        probe = object.__new__(ComparisonRunner)
        probe.cfg, probe.study, probe.exp = cfg, study, EXPERIMENT
        ds, paired = cfg["datasets"][0], {}
        _, adaptations = probe.pair_units(ds)
        for seed in seeds:
            for arm, (_, (label, cond)) in zip(ARMS, adaptations):
                key = probe.unit_key(ds, label, seed)
                unit = load_unit(key, config=None, arrays=False)
                if unit is None:
                    raise ValueError(f"Incomplete or missing unit: {key.run_id}")
                fingerprint = unit.config["data_fingerprint"]
                expected = probe.unit_config({"experiment": EXPERIMENT, "dataset": ds, "seed": seed,
                    "condition": cond, "config": cfg, "data_fingerprint": fingerprint})
                from model.common.tracking import config_identity
                if config_identity(unit.config) != config_identity(expected):
                    raise ValueError(f"Unit configuration mismatch: {key.run_id}")
                events = sorted(records.rows(unit, "retrieval_events"), key=lambda e:e["query_index"])
                row = records.record(unit, "row")
                if [e["query_index"] for e in events] != list(range(int(row["test_n"]))) or any(e["stream"] != "test" for e in events):
                    raise ValueError(f"Incomplete/duplicated query events: {key.run_id}")
                classification = ds["task_type"] == "classification"
                pre = np.array([e["p0"] if classification else e["pre_adaptation_prediction"] for e in events])
                post = np.array([e["final_values"] if classification else e["final_prediction"] for e in events])
                target = np.array([e["target"] for e in events])
                if not all(np.isfinite(x).all() for x in (pre, post, target)):
                    raise ValueError("Nonfinite recorded prediction")
                if arm == "existing":
                    paired[seed] = (fingerprint, target)
                elif fingerprint != paired[seed][0] or not np.array_equal(target, paired[seed][1]):
                    raise ValueError("Old/new query pairing mismatch")
                fn = classification_diagnostics if classification else regression_diagnostics
                summary, detail, scaling = fn(pre, post, target)
                for stage in ("pre", "post"):
                    expected_error = 1-row[f"test_accuracy_{stage}"] if classification else row[f"test_rmse_{stage}"]
                    if not np.isclose(summary[stage+"_error"], expected_error, atol=2e-6, rtol=1e-5):
                        raise ValueError(f"Events do not reproduce {stage} metric of {key.run_id}")
                identity = dict(task=task, task_type=ds["task_type"], seed=seed, arm=arm)
                summary.update(identity, unit_id=key.run_id, data_fingerprint=fingerprint,
                    selected_epoch=row.get("sync_best_epoch", row.get("adapter_best_epoch")),
                    validation_pre=row.get("val_accuracy_pre", row.get("val_rmse_pre")),
                    validation_post=row.get("val_accuracy_post", row.get("val_rmse_post")))
                units.append(summary)
                for i, diagnostic in enumerate(detail):
                    diagnostic.update(identity, query_index=i)
                    if arm == "magnitude_direction":
                        event = events[i]
                        measured = float(np.linalg.norm(post[i]-pre[i]))
                        if not np.isclose(measured, event["magnitude"], atol=2e-6, rtol=1e-5):
                            raise ValueError("recorded magnitude does not equal applied norm")
                        diagnostic.update(magnitude=event["magnitude"],
                            raw_proposal_norm=float(np.linalg.norm(event["raw_proposal"])),
                            **dict(zip(("d_min", "d_mean", "concentration", "disagreement"), event["neighborhood_statistics"])))
                    cases.append(diagnostic)
                curves.extend({**identity, **s} for s in scaling)
                hist = records.record(unit, "history")
                if "sync" in hist:
                    for h in hist["sync"]:
                        histories.append({**identity, "epoch": h["epoch"], "phase": h["phase"],
                            "selected": h["epoch"] == summary["selected_epoch"],
                            "val_L_post": h["val_L_post"], "val_err_post": h["val_err_post"],
                            "train_correction_norm": h["correction_norm"],
                            "train_raw_proposal_norm": h.get("raw_proposal_norm"),
                            "train_predicted_magnitude": h.get("predicted_magnitude")})
                provenance.append({**identity, "unit_id": key.run_id, "data_fingerprint": fingerprint,
                                   "events": len(events), "event_metrics_reproduced": True})
            print(f"[read] {task} s{seed}", flush=True)
    # Task means preserve equal seed weight; event proportions below pool counts.
    task_rows = []
    for task in configs:
        for arm in ARMS:
            selected = [u for u in units if u["task"] == task and u["arm"] == arm]
            total = {"task": task, "task_type": selected[0]["task_type"], "arm": arm,
                     "n_seeds": len(selected), "n": sum(u["n"] for u in selected)}
            for key in selected[0]:
                if key in total or key in {"unit_id", "data_fingerprint", "seed", "selected_epoch", "validation_pre", "validation_post"}:
                    continue
                if key.startswith("count_") or key in {"pre_wrong", "post_wrong", "margin_improves", "margin_worsens", "improved", "worsened"} or key.startswith("unfixed_"):
                    total[key] = sum(u[key] for u in selected)
                elif key.endswith("_sse"):
                    total[key] = sum(u[key] for u in selected)
                else:
                    total[key] = float(np.mean([u[key] for u in selected]))
            for key in list(total):
                if key.startswith("count_"):
                    total["pct_"+key[6:]] = 100*total[key]/total["n"]
            total["selected_epochs"] = ",".join(str(u["selected_epoch"]) for u in selected)
            task_rows.append(total)
    decompositions = []
    for task in configs:
        a, b = [next(r for r in task_rows if r["task"] == task and r["arm"] == arm) for arm in ARMS]
        metric = "error" if a["task_type"] == "classification" else "mse"
        core = b["pre_"+metric]-a["pre_"+metric]
        correction = (b["post_"+metric]-b["pre_"+metric])-(a["post_"+metric]-a["pre_"+metric])
        total = b["post_"+metric]-a["post_"+metric]
        if not np.isclose(core+correction, total, atol=1e-12):
            raise AssertionError("old/new error decomposition mismatch")
        decompositions.append(dict(task=task, metric=metric, core_change=core,
                                   adaptation_effect_change=correction, final_change=total))
    family_rows, scale_rows, task_scales = [], [], []
    for task in configs:
        for arm in ARMS:
            task_cases = [r for r in cases if r["task"] == task and r["arm"] == arm]
            extra = {"task": task, "arm": arm}
            if task.startswith("classification"):
                missing = [r["min_scale_to_correct"] for r in task_cases if r["category"] == "needs_larger_scale"]
                extra.update(n_needs_larger=len(missing), median_required_multiplier=float(np.median(missing)) if missing else None,
                             max_required_multiplier=float(max(missing)) if missing else None)
            if arm == "magnitude_direction":
                magnitudes = np.array([r["magnitude"] for r in task_cases])
                norms = np.array([r["raw_proposal_norm"] for r in task_cases])
                extra.update(magnitude_mean=float(magnitudes.mean()), magnitude_median=float(np.median(magnitudes)),
                    pct_scale_within_20pct_of_initial=float(100*np.mean(abs(magnitudes-.01) <= .002)),
                    median_scale_over_raw_norm=float(np.median(magnitudes[norms > EPS]/norms[norms > EPS])) if np.any(norms > EPS) else None)
            task_scales.append(extra)
    for kind in ("classification", "regression"):
        family_tasks = [t for t in configs if t.startswith(kind)]
        for arm in ARMS:
            ts = [r for r in task_rows if r["task_type"] == kind and r["arm"] == arm]
            fs = {"task_type": kind, "arm": arm, "n": sum(r["n"] for r in ts), "n_tasks": len(ts)}
            for key in ts[0]:
                if key.startswith("count_") or key in {"pre_wrong", "post_wrong"} or key.startswith("unfixed_") or key.endswith("_sse"):
                    fs[key] = sum(r[key] for r in ts)
                if key.startswith("pct_"):
                    fs["equal_task_mean_"+key] = float(np.mean([r[key] for r in ts]))
            family_rows.append(fs)
            for scale in SCALES:
                task_errors = []
                reference = []
                for task in family_tasks:
                    task_errors.append(float(np.mean([r["error"] for r in curves if r["task"] == task and r["arm"] == arm and r["scale"] == scale])))
                    reference.append(next(r["post_error"] for r in ts if r["task"] == task))
                valid = [i for i in range(len(reference)) if task_errors[i] > 0 and reference[i] > 0]
                scale_rows.append({"task_type": kind, "arm": arm, "scale": scale,
                    "mean_task_error": float(np.mean(task_errors)),
                    "gmr_vs_actual": float(np.exp(np.mean([np.log(task_errors[i]/reference[i]) for i in valid]))) if valid else None,
                    "zero_error_exclusions": ",".join(family_tasks[i] for i in range(len(reference)) if i not in valid)})
    return dict(study=study, seeds=list(seeds), n_tasks=len(configs), n_units=len(units),
                n_event_evaluations=len(cases), tolerance=EPS, scales=list(SCALES),
                units=units, tasks=task_rows, cases=cases, curves=curves, histories=histories,
                decompositions=decompositions, provenance=provenance, families=family_rows,
                scaling_summary=scale_rows, task_scales=task_scales)


def plots(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    specifications = [
        ("regression", [("toward_target_short", "Toward target, short", "#3b82a0"),
                        ("overshoot_still_helps", "Overshoot, still helps", "#8b80be"),
                        ("wrong_direction", "Wrong direction", "#e09b42"),
                        ("overshoot_harms", "Overshoot, harms", "#bf5263"),
                        ("moved_from_exact", "Moved from an accurate retrieval", "#bbc1c8")], "Regression: direction and size of each correction", "All test evaluations (%)"),
        ("classification", [("fixed", "Fixed", "#3b82a0"),
                            ("needs_larger_scale", "Too small along current direction", "#8b80be"),
                            ("needs_smaller_scale", "Too large along current direction", "#bf5263"),
                            ("no_positive_scale_rescues", "No positive scale can fix", "#e09b42")],
         "Classification: what happened to retrieval errors?", "Retrieval errors (%)")]
    for kind, categories, title, xlabel in specifications:
        rows = [r for r in result["tasks"] if r["task_type"] == kind]
        fig, ax = plt.subplots(figsize=(12, 8.2))
        y, labels = np.arange(len(rows)), []
        left = np.zeros(len(rows))
        denom = np.array([r["n"] if kind == "regression" else r["pre_wrong"] for r in rows])
        for key, label, color in categories:
            count = np.array([r["count_"+key] for r in rows])
            values = 100*np.divide(count, denom, out=np.zeros(len(rows)), where=denom > 0)
            ax.barh(y, values, left=left, color=color, label=label, height=.72)
            for yy, l, value in zip(y, left, values):
                if value >= 7:
                    ax.text(l+value/2, yy, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=9)
            left += values
        remainder = 100-left
        if np.any(remainder > .01):
            ax.barh(y, remainder, left=left, color="#e2e5e8", height=.72, label="Unchanged / boundary cases")
        for r, n in zip(rows, denom):
            name = r["task"].split("__")[1].replace("_", " ")
            labels.append(("diabetes" if name == "diabets" else name) + (" | old" if r["arm"] == "existing" else " | new") + f" (n={int(n)})")
        ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel=xlabel, title=title)
        ax.invert_yaxis()
        for pos in np.arange(1.5, len(rows), 2):
            ax.axhline(pos, color="#dddddd", lw=.7)
        ax.legend(loc="upper center", bbox_to_anchor=(.47, -.09), ncol=2, frameon=False, fontsize=9)
        seed_text = ", ".join(str(s) for s in result["seeds"])
        fig.text(.02, .012, f"Study {result['study']}; seeds {seed_text} pooled within each task. Hindsight diagnostics on each adapter's own selected retrieval core.", fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, 1))
        fig.savefig(out / f"{kind}_behavior.png", dpi=170)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = (args.out or REPO/"results"/"adaptation_improvement"/args.study/"behavior").resolve()
    if not out.is_relative_to((REPO/"results").resolve()):
        parser.error("output must remain under this worktree's results/")
    result = analyze(args.study, args.tasks)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("units", "tasks", "cases", "curves", "histories", "decompositions", "provenance", "families", "scaling_summary", "task_scales"):
        write_csv(out/(name+".csv"), result[name])
    compact = {k:v for k,v in result.items() if k not in ("cases", "histories")}
    (out/"diagnostics.json").write_text(json.dumps(compact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plots(result, out)
    print(json.dumps({"tasks": result["n_tasks"], "units": result["n_units"],
                      "event_evaluations": result["n_event_evaluations"], "output": str(out)}))
    return result


if __name__ == "__main__":
    main()
