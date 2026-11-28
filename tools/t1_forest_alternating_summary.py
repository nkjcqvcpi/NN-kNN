"""Read-only checks and tables for the alternating forest study."""
import numpy as np

from model.t1 import records
from model.t1.forest_alternating import neighborhood_drift
from tools import t1_forest_alternating as d
from tools import t1_pairwise_redesign_summary as old


def load_family(study, task, seed, family, context_unit, initial_context, core_unit, candidates, cycles, checks):
    source = d.source_family(task, seed, family, context_unit)
    forest, ctx_unit = source["candidate"], context_unit
    context = initial_context
    retrievals, refits = [], []
    baseline = records.record(source["candidate"], "row")
    cycles.append(dict(task=task, task_type=initial_context["task_type"], seed=seed, family=family,
        cycle=0, val_error=baseline["val_error"], val_task_loss=baseline["val_task_loss"],
        val_pre_error=d.metrics(context["val"]["p0"], context["val"]["target"], context["task_type"])["error"],
        core_hash=initial_context["core_hash"], retrieval_updates=0))
    for cycle in range(1, d.PROTOCOL["cycles"] + 1):
        unit = old.must_load(d.key_for(study, task, seed, f"r{d.TAGS[family]}{cycle}"),
            d.config_for(task, seed, family, cycle, "retrieval", source,
                core_source=core_unit, forest_source=forest, context_source=ctx_unit))
        row = records.record(unit, "row")
        updated = d.p.numpy_state(unit.load_state("context"))
        if "test" in updated:
            raise AssertionError("retrieval/refit context should not contain test queries")
        actual_hash = d.p.s.state_hash(unit.load_state("core_state"))
        if actual_hash != updated["core_hash"] or actual_hash != row["core_hash"] or row["previous_core_hash"] != context["core_hash"]:
            raise AssertionError("alternating core checkpoint chain differs")
        for split in ("train", "val"):
            np.testing.assert_array_equal(updated[split]["target"], initial_context[split]["target"])
        np.testing.assert_array_equal(updated["case_targets"], initial_context["case_targets"])
        for key, value in neighborhood_drift(context["train"], updated["train"]).items():
            if not np.isclose(value, row[key], rtol=1e-12, atol=1e-12):
                raise AssertionError("neighborhood-change diagnostic differs")
        checks["alternating_retrieval_states"] += 1
        checks["alternating_changed_core_states"] += int(actual_hash != row["previous_core_hash"])
        history = records.record(unit, "history")
        if len(history) != d.PROTOCOL["retrieval_epochs_per_cycle"] or row["retrieval_updates"] != sum(h["retrieval_updates"] for h in history):
            raise AssertionError("retrieval update budget differs")
        checks["alternating_epochs_with_nonzero_post_gradient_probe"] += sum(h["post_gradient_norm_first_batch"] > 0 for h in history)
        fitted = old.must_load(d.key_for(study, task, seed, f"f{d.TAGS[family]}{cycle}"),
            d.config_for(task, seed, family, cycle, "refit", source, retrieval_source=unit))
        fit_row = records.record(fitted, "row")
        if any(k.startswith("test_") for k in fit_row) or fit_row["core_hash"] != actual_hash:
            raise AssertionError("forest refit has test metrics or wrong core")
        pairs = d.training_pairs(updated, family, seed)
        if d.rf.pair_digest(pairs) != fit_row["pair_digest"]:
            raise AssertionError("refreshed pairs differ from saved training digest")
        model = records.unpack(fitted.load_state("model"))
        expected_type = ("RandomForest" if "random_forest" in family else "ExtraTrees") + (
            "Classifier" if updated["task_type"] == "classification" else "Regressor")
        if type(model).__name__ != expected_type:
            raise AssertionError("forest estimator changed")
        for name in ("n_estimators", "min_samples_leaf", "max_depth", "max_features", "bootstrap"):
            if model.get_params()[name] != source["settings"][name]:
                raise AssertionError("forest settings differ from matched sequential model")
        if model.random_state != seed:
            raise AssertionError("forest random seed changed")
        saved = d.p.numpy_state(fitted.load_state("val_predictions"))
        old.validate_prediction(saved, updated["val"], updated["task_type"], fit_row, "val_", frozen=True, probabilities=True)
        candidates.append(dict(fit_row, study=study, candidate=f"{d.TAGS[family]}_{cycle}"))
        cycles.append(dict(row, **{k: v for k, v in fit_row.items() if k.startswith("val_")},
            refit_unit=fitted.info.run_id, post_gradient_probe_mean=float(np.mean([h["post_gradient_norm_first_batch"] for h in history])),
            train_pre_loss_start=history[0]["train_pre_loss"], train_pre_loss_end=history[-1]["train_pre_loss"],
            train_final_loss_start=history[0]["train_final_loss"], train_final_loss_end=history[-1]["train_final_loss"]))
        checks["alternating_refit_validation_predictions"] += 1
        checks["alternating_pair_digest_and_estimator_checks"] += 1
        retrievals.append(unit)
        refits.append(fitted)
        core_unit, forest, ctx_unit, context = unit, fitted, unit, updated
    config = d.selected_config(task, seed, family, source, retrievals, refits)
    unit = old.must_load(d.key_for(study, task, seed, "e" + d.TAGS[family]), config)
    row = records.record(unit, "row")
    cycle = config["selected_cycle"]
    chosen = d.choose_checkpoint(d.checkpoint_rows(source, refits))
    if row["selected_cycle"] != cycle or row["initial_core_hash"] != initial_context["core_hash"]:
        raise AssertionError("selected checkpoint differs")
    expected_hash = initial_context["core_hash"] if cycle == 0 else records.record(retrievals[cycle - 1], "row")["core_hash"]
    if row["selected_core_hash"] != expected_hash:
        raise AssertionError("selected core hash differs")
    for name in ("val_error", "val_task_loss"):
        if row[name] != chosen[name]:
            raise AssertionError("selection is not validation-only")
    saved = d.p.numpy_state(unit.load_state("test_predictions"))
    part = d.p.numpy_state(unit.load_state("test_context"))
    np.testing.assert_array_equal(part["target"], initial_context["test"]["target"])
    old.validate_prediction(saved, part, initial_context["task_type"], row, "test_", frozen=True, probabilities=True)
    if cycle == 0:
        old_saved = d.p.numpy_state(source["selected"].load_state("test_predictions"))
        for name in saved:
            np.testing.assert_array_equal(saved[name], old_saved[name])
    for name, value in neighborhood_drift(initial_context["test"], part).items():
        if not np.isclose(row[name], value, rtol=1e-12, atol=1e-12):
            raise AssertionError("selected test neighborhood diagnostic differs")
    checks["alternating_selected_predictions_and_choices"] += 1
    checks["alternating_selected_cycle_zero"] += int(cycle == 0)
    return dict(row, study=study), saved


def aggregate_diagnostics(rows, cycles):
    summary = []
    for kind in ("classification", "regression"):
        for family in d.FAMILIES:
            group = [r for r in rows if r["task_type"] == kind and r["method"] == family + "_alternating"]
            if not group:
                continue
            by_task = sorted({r["task"] for r in group})
            mean = lambda key: float(np.mean([np.mean([r[key] for r in group if r["task"] == task]) for task in by_task]))
            trajectory = [r for r in cycles if r["task_type"] == kind and r["family"] == family]
            initial = {(r["task"], r["seed"]): r for r in trajectory if r["cycle"] == 0}
            last = [r for r in trajectory if r["cycle"] == d.PROTOCOL["cycles"]]
            summary.append(dict(task_type=kind, family=family, n_runs=len(group),
                cycle_zero=sum(r["selected_cycle"] == 0 for r in group),
                mean_selected_cycle=float(np.mean([r["selected_cycle"] for r in group])),
                mean_initial_error=mean("test_initial_error"), mean_updated_retrieval_error=mean("test_pre_error"),
                mean_final_error=mean("test_error"), mean_neighborhood_changed_fraction=mean("neighborhood_changed_fraction"),
                mean_activation_l1=mean("mean_activation_l1"),
                selected_retrieval_worse_runs=sum(r["test_pre_error"] > r["test_initial_error"] + 1e-8 for r in group),
                last_cycle_validation_worse_runs=sum(r["val_error"] > initial[(r["task"], r["seed"])]["val_error"] + 1e-8 for r in last),
                mean_last_cycle_validation_error_change=float(np.mean([r["val_error"] - initial[(r["task"], r["seed"])]["val_error"] for r in last]))))
    return summary
