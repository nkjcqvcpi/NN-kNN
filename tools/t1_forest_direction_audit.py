"""Independently audit RF corrections from recorded selected-test predictions.

Reads completed run units only. No fitting, inference, or changes to run records.
The rows are query evaluations; observations repeat across split seeds. Regression
targets use each split's training standardization. All diagnostics are hindsight,
not deployable gates or new model-selection criteria.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.t1 import records

TASKS = {
    "classification": ("iris", "wine", "breast_cancer", "balance", "digits", "zebra", "zebra_special"),
    "regression": ("airfoil", "car", "yacht", "energy_efficiency", "student_performance", "body_fat", "diabets"),
}
EPS = 1e-8


def distribution(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), p99=float(np.quantile(x, .99)), maximum=float(x.max())) if len(x) else dict(n=0)


def regression(pre, final, target):
    pre, final, target = (np.asarray(x, dtype=float).reshape(-1) for x in (pre, final, target))
    assert pre.shape == final.shape == target.shape
    need, update = target - pre, final - pre
    before, after = (target - pre) ** 2, (target - final) ** 2
    change = after - before
    active, needed = abs(update) > EPS, abs(need) > EPS
    # Independent sign test, without the report's update/residual ratio helper.
    wrong = active & needed & (np.sign(update) != np.sign(need))
    overshoot = active & needed & (np.sign(update) == np.sign(need)) & (abs(update) > (2 + EPS) * abs(need))
    improve, worsen = change < -EPS, change > EPS
    np.testing.assert_allclose(change, update ** 2 - 2 * need * update, rtol=1e-10, atol=1e-12)
    assert np.all(change[wrong] > 0)
    harm, benefit = np.maximum(change, 0), np.maximum(-change, 0)
    result = dict(n=len(pre), wrong_direction=int(wrong.sum()), wrong_direction_pct=float(100 * wrong.mean()),
        inactive=int((~active).sum()), wrong_among_active_pct=float(100 * wrong.sum() / active.sum()),
        wrong_with_sse_increase_over_epsilon=int((wrong & worsen).sum()),
        harmful_overshoot=int(overshoot.sum()), improved=int(improve.sum()), worsened=int(worsen.sum()),
        unchanged_error=int((~improve & ~worsen).sum()), pre_sse=float(before.sum()), post_sse=float(after.sum()),
        pooled_pre_rmse=float(np.sqrt(before.mean())), pooled_final_rmse=float(np.sqrt(after.mean())),
        wrong_direction_harm_sse=float(change[wrong].sum()), total_harm_sse=float(harm.sum()),
        total_benefit_sse=float(benefit.sum()), wrong_share_of_harm_pct=float(100 * change[wrong].sum() / harm.sum()),
        wrong_abs_update=distribution(abs(update[wrong])), helpful_abs_update=distribution(abs(update[improve])),
        wrong_squared_error_increase=distribution(change[wrong]), helpful_squared_error_decrease=distribution(-change[improve]),
        wrong_absolute_error_increase=distribution(abs(target[wrong] - final[wrong]) - abs(need[wrong])),
        wrong_initial_absolute_error=distribution(abs(need[wrong])))
    result["correction_thresholds"] = [dict(threshold=t, wrong_remaining=int((wrong & (abs(update) > t)).sum()),
        wrong_pct_all_queries=float(100 * (wrong & (abs(update) > t)).mean()),
        pct_wrong_retained=float(100 * (wrong & (abs(update) > t)).sum() / wrong.sum()),
        wrong_harm_retained_pct=float(100 * change[wrong & (abs(update) > t)].sum() / change[wrong].sum()))
        for t in (EPS, 1e-6, 1e-4, .001, .01, .05, .1)]
    ordered_harm = np.sort(change[wrong])[::-1]
    result["wrong_harm_concentration"] = [dict(top_fraction=f, n=int(np.ceil(f * len(ordered_harm))),
        share_pct=float(100 * ordered_harm[:int(np.ceil(f * len(ordered_harm)))].sum() / ordered_harm.sum()))
        for f in (.01, .05, .1)]
    np.testing.assert_allclose(before.sum() - after.sum(), benefit.sum() - harm.sum(), rtol=1e-10, atol=1e-12)
    return result


def classification(pre, final, target):
    pre, final = (np.asarray(x, dtype=float) for x in (pre, final))
    target = np.asarray(target, dtype=int).reshape(-1)
    assert pre.shape == final.shape and len(pre) == len(target)
    for x in (pre, final):
        assert x.min() >= 0
        np.testing.assert_allclose(x.sum(1), 1, atol=2e-6, rtol=0)
    pred0, pred1 = pre.argmax(1), final.argmax(1)
    correct0, correct1 = pred0 == target, pred1 == target
    fixed, harmed = ~correct0 & correct1, correct0 & ~correct1
    unchanged = pred0 == pred1
    indices = np.arange(len(target))
    p0, p1 = pre[indices, target], final[indices, target]
    lower = p1 < p0 - EPS
    nll0, nll1 = -np.log(np.clip(p0, 1e-12, 1)), -np.log(np.clip(p1, 1e-12, 1))
    other0, other1 = pre.copy(), final.copy()
    other0[indices, target] = other1[indices, target] = -np.inf
    margin0, margin1 = p0 - other0.max(1), p1 - other1.max(1)
    result = dict(n=len(target), pre_wrong=int((~correct0).sum()), post_wrong=int((~correct1).sum()),
        fixed=int(fixed.sum()), harmful_flip=int(harmed.sum()), harmful_flip_pct=float(100 * harmed.mean()),
        retained_correct=int((correct0 & correct1).sum()), both_wrong=int((~correct0 & ~correct1).sum()),
        unchanged_label=int(unchanged.sum()), unchanged_label_pct=float(100 * unchanged.mean()),
        wrong_to_different_wrong=int((~correct0 & ~correct1 & ~unchanged).sum()),
        true_probability_decreased=int(lower.sum()), true_probability_decreased_pct=float(100 * lower.mean()),
        true_probability_decreased_same_label=int((lower & unchanged).sum()),
        true_probability_decreased_harmful_flip=int((lower & harmed).sum()),
        true_margin_decreased=int((margin1 < margin0 - EPS).sum()),
        pooled_pre_error_pct=float(100 * (~correct0).mean()), pooled_final_error_pct=float(100 * (~correct1).mean()),
        mean_pre_nll=float(nll0.mean()), mean_final_nll=float(nll1.mean()),
        true_probability_drop=distribution((p0 - p1)[lower]))
    assert result["post_wrong"] == result["pre_wrong"] - result["fixed"] + result["harmful_flip"]
    assert result["n"] == result["retained_correct"] + result["fixed"] + result["harmful_flip"] + result["both_wrong"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-study", required=True)
    parser.add_argument("--alternating-study", required=True)
    args = parser.parse_args()
    out = REPO / "results" / "adaptation_improvement" / args.alternating_study
    grouped, per_run, sources = defaultdict(list), [], []
    seen_targets, seen_pre = {}, {}
    for scheme, study, experiment in (("sequential", args.sequential_study, "rf_v1"),
                                       ("alternating", args.alternating_study, "af_v1")):
        for recipe, family, step in (("mixed", "pair_random_forest", "e_m" if scheme == "sequential" else "erm"),
                                    ("retrieved_only", "local_pair_random_forest", "e_l" if scheme == "sequential" else "erl")):
            method = family + ("_alternating" if scheme == "alternating" else "")
            for kind, datasets in TASKS.items():
                for dataset in datasets:
                    for seed in (40, 41, 42):
                        key = records.key(study, experiment, dataset, step, f"s{seed}")
                        unit = tracking.load_unit(key, config=None, arrays=False)
                        if unit is None:
                            raise ValueError(f"Missing completed unit: {key.run_id}")
                        config = unit.config
                        assert (config["family"], config["task"], config["seed"]) == (family, kind + "__" + dataset, seed)
                        saved = unit.load_state("test_predictions")
                        arrays = {k: np.asarray(saved[k].numpy(), dtype=float) for k in ("p0", "final", "target")}
                        assert all(np.isfinite(v).all() for v in arrays.values())
                        split = kind, dataset, seed
                        if split in seen_targets:
                            np.testing.assert_array_equal(arrays["target"], seen_targets[split])
                        seen_targets[split] = arrays["target"]
                        if scheme == "sequential":
                            if split in seen_pre:
                                np.testing.assert_array_equal(arrays["p0"], seen_pre[split])
                            seen_pre[split] = arrays["p0"]
                        compute = regression if kind == "regression" else classification
                        diag = compute(arrays["p0"], arrays["final"], arrays["target"])
                        row = records.record(unit, "row")
                        actual = diag["pooled_final_rmse"] if kind == "regression" else diag["post_wrong"] / diag["n"]
                        np.testing.assert_allclose(actual, row["test_error"], rtol=1e-7, atol=1e-9)
                        grouped[kind, method].append(arrays)
                        per_run.append(dict(task_type=kind, dataset=dataset, seed=seed, scheme=scheme, recipe=recipe,
                                            method=method, unit=key.run_id, diagnostics=diag))
                        sources.append(dict(unit=key.run_id, config_identity=tracking.config_identity(config), state="test_predictions"))
    summaries = []
    for (kind, method), parts in grouped.items():
        # Class counts differ by dataset: concatenate scalar outcomes, not class-score matrices.
        compute = regression if kind == "regression" else None
        if compute is None:
            group = [r["diagnostics"] for r in per_run if (r["task_type"], r["method"]) == (kind, method)]
            count_keys = [k for k, v in group[0].items() if isinstance(v, int)]
            diag = {k: sum(r[k] for r in group) for k in count_keys}
            for name, count in (("harmful_flip_pct", "harmful_flip"), ("unchanged_label_pct", "unchanged_label"),
                                ("true_probability_decreased_pct", "true_probability_decreased"),
                                ("pooled_pre_error_pct", "pre_wrong"), ("pooled_final_error_pct", "post_wrong")):
                diag[name] = 100 * diag[count] / diag["n"]
            for name in ("mean_pre_nll", "mean_final_nll"):
                diag[name] = sum(r[name] * r["n"] for r in group) / diag["n"]
        else:
            arrays = {k: np.concatenate([part[k] for part in parts]) for k in ("p0", "final", "target")}
            diag = compute(arrays["p0"], arrays["final"], arrays["target"])
        summaries.append(dict(task_type=kind, method=method, diagnostics=diag))
    result = dict(scope="Recorded selected RF test predictions; no retraining or model inference",
        seeds=[40, 41, 42], units=len(sources), epsilon=EPS,
        weighting="Pooled query evaluations; repeated observations across seeds; not equal-dataset means",
        regression_units="Each split's training-standardized target units",
        classification_note="True-probability decreases are separate from regression wrong-direction counts",
        summaries=summaries, per_run=per_run, sources=sources)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rf_direction_audit.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact = []
    for row in summaries:
        keys = ("n", "wrong_direction", "wrong_direction_pct", "wrong_with_sse_increase_over_epsilon",
                "wrong_share_of_harm_pct", "wrong_abs_update", "wrong_squared_error_increase",
                "total_benefit_sse", "total_harm_sse", "pooled_pre_rmse", "pooled_final_rmse") if row["task_type"] == "regression" else (
                "n", "pre_wrong", "post_wrong", "fixed", "harmful_flip", "unchanged_label_pct",
                "true_probability_decreased", "true_probability_decreased_same_label")
        compact.append(dict(task_type=row["task_type"], method=row["method"], **{k: row["diagnostics"][k] for k in keys}))
    print(json.dumps(dict(output=str(path), units=len(sources), summaries=compact), indent=2))


if __name__ == "__main__":
    main()
