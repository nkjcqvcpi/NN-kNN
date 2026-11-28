"""Read recorded pairwise experiments; never fit or evaluate a model here."""
from collections import Counter
import json
import numpy as np
from model.common import tracking
from model.t1 import records
from tools import t1_pairwise_redesign as p
from tools.t1_adaptation_diagnostics import classification_diagnostics, regression_diagnostics

s, drop = p.s, p.drop
METHODS = ("retrieval", *p.FAMILIES, "validation_choice", "nncdh_original_seq0",
    "nncdh_original_seq20", "nncdh_seq_selected", "nncdh_all_selected", "nncdh_original_mixed")
LABELS = dict(retrieval="Frozen retrieval", local_ridge="Local ridge", pair_trees="Pairwise trees",
    pair_neural="Pairwise neural", validation_choice="Validation choice (including retrieval)",
    nncdh_original_seq0="Original NN-CDH: sequential, no dropout",
    nncdh_original_seq20="Original NN-CDH: sequential, 20% dropout",
    nncdh_seq_selected="NN-CDH: validation choice of 9 sequential",
    nncdh_all_selected="NN-CDH: validation choice of all 27",
    nncdh_original_mixed="Original NN-CDH: historical mixed scheme")


def must_load(key, config):
    unit = tracking.load_unit(key, config=config, arrays=False)
    if unit is None:
        raise ValueError(f"Missing completed run: {key.run_id}")
    return unit


def choose_policy(rows):
    order = ("retrieval", *p.FAMILIES)
    return min(rows, key=lambda r: (r["val_error"], order.index(r["method"])))


def choose_nncdh(rows):
    return min(rows, key=lambda r: (r["val_task_loss"], r["val_error"], r["dropout"],
        list(s.DESIGNS).index(r["adapter"]), s.SCHEMES.index(r["scheme"])))


def nncdh_references(task, seed, refs, sources, context):
    """Retain previous selection criteria, including original mixed-scheme results."""
    base = records.unpack(refs[2].load_state("objects"))
    if s.state_hash(base["model"].state_dict()) != context["core_hash"]:
        raise AssertionError("source core hash differs")
    desc = sources["original"].load_state("checkpoint")["model_description"]
    bs = desc.get("adapter", {}).get("batch_size", base["cc"].batch_size)
    batches = (len(context["train"]["target"]) + bs - 1) // bs
    del base
    rows, saved_by_unit, units = [], {}, {}
    for adapter in s.DESIGNS:
        for scheme in s.SCHEMES:
            zero = drop.zero_unit(task, seed, adapter, scheme, refs, sources)
            for rate in drop.RATES:
                if rate == 0:
                    unit = zero
                    if scheme == s.reused_scheme(task):
                        row, saved = s.reused_row(task, seed, adapter, unit, sources["query_aware"],
                            batches, context["core_hash"])
                        saved_by_unit[unit.info.run_id] = saved
                    else:
                        row, = records.rows(unit, "rows")
                else:
                    unit = must_load(drop.key_for("ad1", task, seed, adapter, scheme, rate),
                        drop.config_for(task, seed, adapter, scheme, rate, refs, zero))
                    row, = records.rows(unit, "rows")
                row = dict(row, dropout=rate, adapter=adapter, scheme=scheme, unit=unit.info.run_id)
                rows.append(row)
                units[unit.info.run_id] = unit
    sequential = [r for r in rows if r["scheme"] == "sequential"]
    def fixed(rate):
        return next(r for r in sequential if r["adapter"] == "original" and r["dropout"] == rate)
    chosen = dict(nncdh_original_seq0=fixed(0.), nncdh_original_seq20=fixed(.2),
        nncdh_seq_selected=choose_nncdh(sequential), nncdh_all_selected=choose_nncdh(rows),
        nncdh_original_mixed=next(r for r in rows if r["adapter"] == "original" and
            r["scheme"] == s.reused_scheme(task) and r["dropout"] == 0.))
    result = []
    for method, row in chosen.items():
        unit_id = row["unit"]
        saved = saved_by_unit.get(unit_id)
        if saved is None:
            saved = units[unit_id].load_state("test_predictions")
        result.append((dict(row, method=method, selected_method=row["method"]), p.numpy_state(saved)))
    return result


def validate_prediction(saved, part, kind, row, prefix, *, frozen, probabilities):
    if not np.array_equal(saved["target"], part["target"]):
        raise AssertionError("prediction targets do not match source split")
    if frozen and not np.array_equal(saved["p0"], part["p0"]):
        raise AssertionError("frozen predictions differ")
    final = saved["final"]
    if not np.isfinite(final).all():
        raise AssertionError("nonfinite prediction")
    if probabilities and kind == "classification":
        if (final < -1e-12).any() or not np.allclose(final.sum(1), 1., atol=2e-6, rtol=0):
            raise AssertionError("invalid probability distribution")
    measured = p.metrics(final, saved["target"], kind, saved["p0"])
    if not np.isclose(measured["error"], row[prefix + "error"], atol=2e-6, rtol=1e-5):
        raise AssertionError("reported error differs from saved predictions")
    if probabilities and not np.isclose(measured["task_loss"], row[prefix + "task_loss"], atol=1e-9, rtol=1e-7):
        raise AssertionError("reported final loss differs from saved predictions")


def add_behavior(row, saved, behavior, harms):
    kind = row["task_type"]
    pre, post, target = (np.asarray(saved[k]) for k in ("p0", "final", "target"))
    if kind == "regression":
        pre, post, target = (v.reshape(-1) for v in (pre, post, target))
        stats, cases, _ = regression_diagnostics(pre, post, target)
    else:
        stats, cases, _ = classification_diagnostics(pre, post, target)
    identity = {k: row[k] for k in ("task", "task_type", "seed", "method", "unit")}
    stats = {k: v for k, v in stats.items() if not k.startswith("oracle_")}
    behavior.append(dict(**identity, **stats))
    if kind == "regression":
        worst = sorted(enumerate(cases), key=lambda x: x[1]["squared_error_change"], reverse=True)[:5]
        for index, case in worst:
            if case["squared_error_change"] > 0:
                harms.append(dict(**identity, test_index=index, **case,
                    fraction_of_run_post_sse=case["post_squared_error"] / (stats["post_mse"] * stats["n"])))


def behavior_totals(behavior, methods=METHODS):
    totals = []
    for kind in ("classification", "regression"):
        for method in methods:
            group = [r for r in behavior if r["task_type"] == kind and r["method"] == method]
            if not group:
                continue
            keys = [k for k in group[0] if k.startswith("count_") or k.endswith("_sse") or
                k in ("n", "pre_wrong", "post_wrong", "improved", "worsened", "margin_improves", "margin_worsens")]
            row = dict(task_type=kind, method=method, **{k: sum(r[k] for r in group) for k in keys})
            row["pooled_pre_error"] = sum(r["pre_error"] * r["n"] for r in group) / row["n"]
            row["pooled_post_error"] = sum(r["post_error"] * r["n"] for r in group) / row["n"]
            if kind == "regression":
                for part in ("pre", "post"):
                    row["pooled_" + part + "_error"] = float(np.sqrt(sum(r[part + "_mse"] * r["n"] for r in group) / row["n"]))
            totals.append(row)
    return totals


def plot_results(out, families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, kind in zip(axes, ("classification", "regression")):
        values = {r["method"]: r["mean_error"] for r in families if r["task_type"] == kind}
        if not values:
            ax.set_visible(False)
            continue
        factor = 100 if kind == "classification" else 1
        bars = [values[m] * factor for m in METHODS]
        colors = ["#153C57" if m == "validation_choice" else "#3487B5" if m in p.FAMILIES else "#A8AFB5" for m in METHODS]
        ax.barh(range(len(METHODS)), bars, color=colors, height=.68)
        ax.set_title(kind.title() + " — lower is better", loc="left", fontsize=12)
        ax.set_xlabel("Mean classification error (%)" if factor == 100 else "Mean RMSE (standardized targets)")
        ax.set_xlim(0, max(bars) * 1.19)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(bars):
            ax.text(value + max(bars) * .015, i, f"{value:.2f}" if factor == 100 else f"{value:.4f}", va="center", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Pairwise adapter redesign: recorded local comparison", x=.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(.01, .015, "Equal dataset weights; 3 seeds per task. NN-CDH all-27 and historical references may change retrieval.\nValidation choices use no test labels. Search budgets and training objectives differ.", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def summarize(study, tasks, seeds):
    rows, candidates, selections, behavior, harms = [], [], [], [], []
    checks = Counter()
    for task in tasks:
        kind = task.split("__")[0]
        for seed in seeds:
            refs, sources = s.references(task, seed)
            context_unit = must_load(p.key_for(study, task, seed, "ctx"), p.context_config(task, seed, refs))
            context = p.numpy_state(context_unit.load_state("context"))
            checks["contexts"] += 1
            test, val = context["test"], context["val"]
            retrieval = dict(task=task, task_type=kind, seed=seed, method="retrieval", adapter="none",
                scheme="pretrained", unit=context_unit.info.run_id, selected_method="retrieval",
                **{"val_" + k: v for k, v in p.metrics(val["p0"], val["target"], kind).items()},
                **{"test_" + k: v for k, v in p.metrics(test["p0"], test["target"], kind).items()})
            split_results = [(retrieval, dict(p0=test["p0"], final=test["p0"], target=test["target"]))]
            for family in p.FAMILIES:
                units = []
                for cid in p.CANDIDATES[family]:
                    unit = must_load(p.key_for(study, task, seed, cid), p.candidate_config(task, seed, family, cid, context_unit))
                    row = records.record(unit, "row")
                    if any(k.startswith("test_") for k in row):
                        raise AssertionError("candidate test results must not be used in selection")
                    saved = p.numpy_state(unit.load_state("val_predictions"))
                    validate_prediction(saved, val, kind, row, "val_", frozen=True, probabilities=True)
                    candidates.append(row)
                    units.append(unit)
                    checks["candidate_validation_predictions"] += 1
                chosen = p.choose_candidate([records.record(u, "row") for u in units])
                selected = must_load(p.key_for(study, task, seed, "eval_" + family.split("_")[-1]),
                    p.selected_config(task, seed, family, units, chosen))
                row = records.record(selected, "row")
                if row["candidate"] != chosen["candidate"]:
                    raise AssertionError("selection differs from validation rule")
                if row["initial_core_hash"] != context["core_hash"] or row["selected_core_hash"] != context["core_hash"]:
                    raise AssertionError("new method changed retrieval")
                saved = p.numpy_state(selected.load_state("test_predictions"))
                validate_prediction(saved, test, kind, row, "test_", frozen=True, probabilities=True)
                checks["selected_test_predictions"] += 1
                checks["frozen_hashes"] += 1
                split_results.append((row, saved))
                selections.append(dict(task=task, task_type=kind, seed=seed, policy=family,
                    selected=chosen["candidate"], val_error=row["val_error"], unit=row["unit"]))
            policy = choose_policy([r for r, _ in split_results])
            policy_saved = next(saved for r, saved in split_results if r["method"] == policy["method"])
            split_results.append((dict(policy, method="validation_choice", selected_method=policy["method"]), policy_saved))
            selections.append(dict(task=task, task_type=kind, seed=seed, policy="validation_choice",
                selected=policy["method"], val_error=policy["val_error"], unit=policy["unit"]))
            for row, saved in nncdh_references(task, seed, refs, sources, context):
                validate_prediction(saved, test, kind, row, "test_", frozen=row["scheme"] == "sequential", probabilities=False)
                checks["nncdh_test_predictions"] += 1
                split_results.append((row, saved))
                selections.append(dict(task=task, task_type=kind, seed=seed, policy=row["method"],
                    selected=f"{row['adapter']}/{row['scheme']}/{row['dropout']}",
                    val_error=row["val_error"], unit=row["unit"]))
            for row, saved in split_results:
                row = dict(row)
                row["test_pre_error"] = p.metrics(saved["p0"], saved["target"], kind)["error"]
                add_behavior(row, saved, behavior, harms)
                rows.append(row)
            print(f"[summary] {task} s{seed}", flush=True)
    if len({(r["task"], r["seed"], r["method"]) for r in rows}) != len(tasks) * len(seeds) * len(METHODS):
        raise AssertionError("incomplete or duplicate result matrix")
    task_rows, families = s.aggregate(rows, tasks, seeds)
    comparisons = []
    for method in (*p.FAMILIES, "validation_choice"):
        for ref in ("retrieval", "nncdh_original_seq0", "nncdh_original_seq20", "nncdh_seq_selected", "nncdh_all_selected"):
            for kind in ("classification", "regression"):
                result = drop.pair_summary(task_rows, rows, method, ref, kind)
                if result is not None:
                    comparisons.append(result)
    selection_counts = [dict(task_type=kind, policy=policy, selected=selected, count=count)
        for (kind, policy, selected), count in sorted(Counter((r["task_type"], r["policy"], r["selected"]) for r in selections).items())]
    totals = behavior_totals(behavior)
    out = p.REPO / "results" / "adaptation_improvement" / study
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(candidates=candidates, rows=rows, tasks=task_rows, families=families,
        comparisons=comparisons, selections=selections, selection_counts=selection_counts,
        behavior=behavior, behavior_totals=totals, harm_examples=harms)
    for name, table in tables.items():
        s.write_csv(out / (name + ".csv"), table)
    payload = dict(study=study, experiment=p.EXPERIMENT, protocol=p.PROTOCOL,
        checks=dict(checks), tasks_requested=list(tasks), seeds=list(seeds), **tables)
    (out / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_results(out, families)
    print(json.dumps(dict(output=str(out), checks=dict(checks), families=families), indent=2), flush=True)
    return 0
