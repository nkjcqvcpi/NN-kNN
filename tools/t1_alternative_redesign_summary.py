"""Merge all recorded alternatives; no model fitting or new inference."""
from collections import Counter
import json
import numpy as np
from model.common import tracking
from model.t1 import records
from tools import t1_pairwise_redesign as p
from tools import t1_pairwise_redesign_summary as old
from tools import t1_alternative_redesign as a

POLICY_ORDER = ("retrieval", "pair_trees", "local_pair_trees", "anchor_transport", "error_memory",
    "vote_attention", "local_ridge", "pair_neural")
METHODS = ("retrieval", *p.FAMILIES, *a.FAMILIES, "ar1_validation_choice", "validation_choice_all",
    "nncdh_original_seq0", "nncdh_original_seq20", "nncdh_seq_selected", "nncdh_all_selected", "nncdh_original_mixed")
LABELS = dict(old.LABELS, direct_trees="Direct ExtraTrees control", local_pair_trees="Pair trees: retrieved-only training",
    anchor_transport="Cross-fitted anchor transport", error_memory="Retrieval-error memory", vote_attention="Neural vote attention",
    ar1_validation_choice="Validation choice: first alternatives", validation_choice_all="Validation choice: all alternatives")
LABELS.update(pair_random_forest="Random Forest: mixed pairs",
    local_pair_random_forest="Random Forest: retrieved pairs",
    validation_choice_with_rf="Validation choice: including RF")


def choose_all(rows, *, include_rf=False):
    order = POLICY_ORDER + (("pair_random_forest", "local_pair_random_forest") if include_rf else ())
    options = [r for r in rows if r["method"] in order]
    return min(options, key=lambda r: (r["val_error"], order.index(r["method"])))


def support_diagnostics(context, part):
    targets = context["case_targets"][part["indices"]]
    valid = part["weights"] > 0
    if context["task_type"] == "classification":
        possible = (targets * valid[..., None]).sum(1) > 0
        missing = ~possible[np.arange(len(possible)), part["target"].astype(int)]
        return dict(n=len(possible), outside_support=int(missing.sum()), oracle_support_error=float(missing.mean()))
    lower = np.where(valid, targets[..., 0], np.inf).min(1)
    upper = np.where(valid, targets[..., 0], -np.inf).max(1)
    y = part["target"].reshape(-1)
    residual = y - np.clip(y, lower, upper)
    return dict(n=len(y), outside_support=int((abs(residual) > 1e-8).sum()),
        oracle_support_error=float(np.sqrt(np.mean(residual ** 2))))


def verify_attention_support(context, part, prediction):
    targets = context["case_targets"][part["indices"]]
    valid = part["weights"] > 0
    if context["task_type"] == "classification":
        absent = (targets * valid[..., None]).sum(1) == 0
        if np.any(prediction[absent] != 0):
            raise AssertionError("attention invented an absent class")
    else:
        low = np.where(valid, targets[..., 0], np.inf).min(1)
        high = np.where(valid, targets[..., 0], -np.inf).max(1)
        tolerance = 1e-6 * np.maximum(1., np.maximum(abs(low), abs(high)))
        value = prediction.reshape(-1)
        if np.any(value < low - tolerance) or np.any(value > high + tolerance):
            raise AssertionError("attention escaped the retrieved target range")


def load_family(driver, study, task, seed, family, context_unit, context, candidate_rows, checks):
    candidates = []
    expected_digest = (driver.pair_digest(driver.training_pairs(context, family, seed))
        if driver.EXPERIMENT == "rf_v1" else None)
    for cid in driver.CANDIDATES[family]:
        unit = old.must_load(driver.key_for(study, task, seed, cid),
            driver.candidate_config(task, seed, family, cid, context_unit))
        row = records.record(unit, "row")
        if expected_digest is not None:
            if row["pair_digest"] != expected_digest:
                raise AssertionError("Random Forest pair recipe differs from the reference builder")
            checks["rf_pair_recipe_matches"] += 1
        if any(k.startswith("test_") for k in row):
            raise AssertionError("unselected candidate has test fields")
        saved = p.numpy_state(unit.load_state("val_predictions"))
        old.validate_prediction(saved, context["val"], context["task_type"], row, "val_", frozen=True, probabilities=True)
        checks[study + "_validation_predictions"] += 1
        if family == "anchor_transport":
            model = records.unpack(unit.load_state("model"))
            for case, fold in enumerate(model["fold_of_case"]):
                if case in model["training_indices"][fold]:
                    raise AssertionError("anchor used to fit its own transport predictor")
            checks["transport_exclusion_sets"] += 1
        if family == "vote_attention":
            verify_attention_support(context, context["val"], saved["final"])
            checks["attention_support_sets"] += 1
        candidates.append(unit)
        candidate_rows.append(dict(row, study=study))
    chosen = p.choose_candidate([records.record(u, "row") for u in candidates])
    evaluation = "eval_" + family.split("_")[-1] if driver is p else "e_" + driver.TAGS[family]
    unit = old.must_load(driver.key_for(study, task, seed, evaluation),
        driver.selected_config(task, seed, family, candidates, chosen))
    row = records.record(unit, "row")
    if row["candidate"] != chosen["candidate"] or row["selected_core_hash"] != context["core_hash"]:
        raise AssertionError("selected candidate/core differs")
    saved = p.numpy_state(unit.load_state("test_predictions"))
    old.validate_prediction(saved, context["test"], context["task_type"], row, "test_", frozen=True, probabilities=True)
    checks[study + "_selected_test_predictions"] += 1
    checks["frozen_hashes"] += 1
    if family == "vote_attention":
        verify_attention_support(context, context["test"], saved["final"])
        checks["attention_support_sets"] += 1
    return dict(row, study=study), saved


def plot_comparison(out, families, *, include_rf=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("retrieval", "nncdh_original_seq20", "local_ridge", "pair_neural", "pair_trees",
        "local_pair_trees", "direct_trees", "anchor_transport", "error_memory", "vote_attention", "validation_choice_all")
    labels = dict(LABELS)
    if include_rf:
        methods = ("retrieval", "nncdh_original_seq20", "local_ridge", "pair_neural",
            "pair_trees", "pair_random_forest", "local_pair_trees", "local_pair_random_forest",
            "direct_trees", "anchor_transport", "error_memory", "vote_attention",
            "validation_choice_all", "validation_choice_with_rf")
        labels.update(pair_trees="ExtraTrees: mixed pairs", local_pair_trees="ExtraTrees: retrieved pairs",
            validation_choice_all="Validation choice: before RF")
    fig, axes = plt.subplots(1, 2, figsize=(15, 8 if include_rf else 7), sharey=True)
    for ax, kind in zip(axes, ("classification", "regression")):
        mapping = {r["method"]: r["mean_error"] for r in families if r["task_type"] == kind}
        if not mapping:
            ax.set_visible(False)
            continue
        factor = 100 if kind == "classification" else 1
        values = [mapping[m] * factor for m in methods]
        colors = ["#909BA2" if m in ("retrieval", "nncdh_original_seq20", "direct_trees") else
            "#173F59" if m == "validation_choice_all" else "#3487B5" for m in methods]
        ax.barh(range(len(methods)), values, color=colors, height=.7)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_title(kind.title() + " — lower is better", loc="left")
        ax.set_xlabel("Mean classification error (%)" if factor == 100 else "Mean RMSE (standardized targets)")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .014, i, f"{value:.2f}" if factor == 100 else f"{value:.4f}", va="center", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(range(len(methods)), [labels[m] for m in methods], fontsize=9)
    axes[0].invert_yaxis()
    title = "Alternatives to NN-CDH: " + ("three" if include_rf else "two") + " completed local studies"
    fig.suptitle(title, x=.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(.01, .015, "14 datasets, 3 seeds; equal dataset weights. All use the same frozen core representations.\nDirect trees bypass retrieval. Attention changes output weights only. Training budgets differ; choices use validation.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(out / "alternatives.png", dpi=160)
    plt.close(fig)


def summarize(study, tasks, seeds, *, forest_study=None, alternating_study=None):
    from tools import t1_pair_random_forest as rf
    from tools import t1_forest_alternating as af
    from tools import t1_forest_alternating_summary as af_summary
    if alternating_study and (study != "ar2" or forest_study != "ar3"):
        raise ValueError("alternating study requires its original ar2/ar3 comparisons")
    methods = METHODS + ((*rf.FAMILIES, "validation_choice_with_rf") if forest_study else ())
    if alternating_study:
        methods += tuple(f + "_alternating" for f in af.FAMILIES)
    rows, candidates, behavior, harms, selections, support = [], [], [], [], [], []
    alternating_cycles = []
    checks = Counter()
    for task in tasks:
        kind = task.split("__")[0]
        for seed in seeds:
            context_unit, context = a.source_context(task, seed)
            checks["contexts"] += 1
            test, val = context["test"], context["val"]
            row = dict(task=task, task_type=kind, seed=seed, method="retrieval", adapter="none", scheme="pretrained",
                unit=context_unit.info.run_id, study="ar1",
                **{"val_" + k: v for k, v in p.metrics(val["p0"], val["target"], kind).items()},
                **{"test_" + k: v for k, v in p.metrics(test["p0"], test["target"], kind).items()})
            split_results = [(row, dict(final=test["p0"], p0=test["p0"], target=test["target"]))]
            for family in p.FAMILIES:
                split_results.append(load_family(p, "ar1", task, seed, family, context_unit, context, candidates, checks))
            previous_choice = old.choose_policy([r for r, _ in split_results])
            previous_saved = next(saved for r, saved in split_results if r["method"] == previous_choice["method"])
            for family in a.FAMILIES:
                split_results.append(load_family(a, study, task, seed, family, context_unit, context, candidates, checks))
            chosen = choose_all([r for r, _ in split_results])
            chosen_saved = next(saved for r, saved in split_results if r["method"] == chosen["method"])
            split_results.extend(((dict(previous_choice, method="ar1_validation_choice", selected_method=previous_choice["method"]), previous_saved),
                (dict(chosen, method="validation_choice_all", selected_method=chosen["method"]), chosen_saved)))
            if forest_study:
                for family in rf.FAMILIES:
                    split_results.append(load_family(rf, forest_study, task, seed, family,
                        context_unit, context, candidates, checks))
                rf_choice = choose_all([r for r, _ in split_results], include_rf=True)
                rf_saved = next(saved for r, saved in split_results if r["method"] == rf_choice["method"])
                split_results.append((dict(rf_choice, method="validation_choice_with_rf",
                    selected_method=rf_choice["method"]), rf_saved))
            refs, sources = p.s.references(task, seed)
            # Validate the original context identity against its source configuration again.
            old.must_load(p.key_for("ar1", task, seed, "ctx"), p.context_config(task, seed, refs))
            if alternating_study:
                for family in af.FAMILIES:
                    split_results.append(af_summary.load_family(alternating_study, task, seed, family,
                        context_unit, context, refs[2], candidates, alternating_cycles, checks))
            for reference, saved in old.nncdh_references(task, seed, refs, sources, context):
                old.validate_prediction(saved, test, kind, reference, "test_", frozen=reference["scheme"] == "sequential", probabilities=False)
                checks["nncdh_test_predictions"] += 1
                split_results.append((reference, saved))
            for row, saved in split_results:
                row = dict(row)
                row["test_pre_error"] = p.metrics(saved["p0"], saved["target"], kind)["error"]
                rows.append(row)
                old.add_behavior(row, saved, behavior, harms)
                selections.append(dict(task=task, task_type=kind, seed=seed, policy=row["method"], unit=row["unit"],
                    selected=row.get("selected_method", row.get("candidate", "retrieval")),
                    val_error=row["val_error"], selected_epoch=row.get("selected_epoch", 0)))
            for part in ("train", "val", "test"):
                support.append(dict(task=task, task_type=kind, seed=seed, split=part, **support_diagnostics(context, context[part])))
            print(f"[summary] {task} s{seed}", flush=True)
    if len({(r["task"], r["seed"], r["method"]) for r in rows}) != len(tasks) * len(seeds) * len(methods):
        raise AssertionError("incomplete or duplicated merged matrix")
    task_rows, families = p.s.aggregate(rows, tasks, seeds)
    comparisons = []
    comparison_methods = (*p.FAMILIES, *a.FAMILIES, "validation_choice_all")
    if forest_study:
        comparison_methods += (*rf.FAMILIES, "validation_choice_with_rf")
    if alternating_study:
        comparison_methods += tuple(f + "_alternating" for f in af.FAMILIES)
    for method in comparison_methods:
        references = ("retrieval", "nncdh_original_seq20", "pair_trees", "direct_trees", "nncdh_all_selected")
        if forest_study and method in (*rf.FAMILIES, "validation_choice_with_rf"):
            references += ("local_pair_trees", "validation_choice_all")
        if method.endswith("_alternating"):
            references = tuple(dict.fromkeys((method.removesuffix("_alternating"), *references)))
        for reference in references:
            if reference == method:
                continue
            for kind in ("classification", "regression"):
                result = p.drop.pair_summary(task_rows, rows, method, reference, kind)
                if result is not None:
                    comparisons.append(result)
    counts = [dict(task_type=kind, policy=policy, selected=selected, count=count)
        for (kind, policy, selected), count in sorted(Counter((r["task_type"], r["policy"], r["selected"]) for r in selections).items())]
    sensitivity = []
    for excluded in ((), ("classification__zebra",), ("classification__zebra", "classification__zebra_special")):
        for method in methods:
            values = [r["test_error"] for r in task_rows if r["method"] == method and r["task_type"] == "classification" and r["task"] not in excluded]
            if values:
                sensitivity.append(dict(method=method, excluded=";".join(excluded), n_tasks=len(values), mean_error=float(np.mean(values))))
    totals = old.behavior_totals(behavior, methods)
    tables = dict(rows=rows, candidates=candidates, tasks=task_rows, families=families, comparisons=comparisons,
        selections=selections, selection_counts=counts, behavior=behavior, behavior_totals=totals,
        harm_examples=harms, support_limits=support, classification_sensitivity=sensitivity)
    if alternating_study:
        tables.update(alternating_cycles=alternating_cycles,
            alternating_diagnostics=af_summary.aggregate_diagnostics(rows, alternating_cycles))
    out = a.REPO / "results" / "adaptation_improvement" / (alternating_study or forest_study or study)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        p.s.write_csv(out / (name + ".csv"), table)
    payload = dict(study=alternating_study or forest_study or study,
        merged_studies=["ar1", study] + ([forest_study] if forest_study else []) + ([alternating_study] if alternating_study else []),
        checks=dict(checks), protocol=af.PROTOCOL if alternating_study else rf.PROTOCOL if forest_study else a.PROTOCOL,
        tasks_requested=list(tasks), seeds=list(seeds), **tables)
    (out / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if alternating_study:
        plot_alternating(out, families)
    else:
        plot_comparison(out, families, include_rf=bool(forest_study))
    print(json.dumps(dict(output=str(out), checks=dict(checks), families=families), indent=2), flush=True)
    return 0


def plot_alternating(out, families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("pair_trees", "local_pair_trees", "pair_random_forest", "local_pair_random_forest")
    labels = ("ExtraTrees: mixed", "ExtraTrees: retrieved", "Random Forest: mixed", "Random Forest: retrieved")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, ("classification", "regression")):
        mapping = {r["method"]: r["mean_error"] for r in families if r["task_type"] == kind}
        if not mapping:
            ax.set_visible(False)
            continue
        scale = 100 if kind == "classification" else 1
        for offset, suffix, label, color in ((-.18, "", "Sequential", "#8a99a5"),
                (.18, "_alternating", "Alternating (validation-selected)", "#17628a")):
            values = [mapping[m + suffix] * scale for m in methods]
            ax.barh(np.arange(4) + offset, values, height=.32, label=label, color=color)
            for i, value in enumerate(values):
                ax.text(value, i + offset, f" {value:.3f}" if scale == 100 else f" {value:.6f}", va="center", fontsize=8)
        ax.set_xlim(0, max(mapping[m + suffix] * scale for m in methods for suffix in ("", "_alternating")) * 1.28)
        ax.set_title(kind.title() + " — lower is better", loc="left")
        ax.set_xlabel("Mean error (%)" if kind == "classification" else "Mean standardized RMSE")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(4), labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(.99, .92),
        ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Does alternating retrieval and forest fitting help?", x=.01, ha="left", weight="bold")
    fig.text(.01, .01, "14 local datasets × 3 seeds; equal dataset weights. Same starting retrieval and forest.\nFive cycles of two retrieval epochs and a forest refit; validation can retain cycle zero. Extra compute and selection; reused splits.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(out / "alternatives.png", dpi=160)
    plt.close(fig)
