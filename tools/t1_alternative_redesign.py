"""Recorded follow-up: creative alternatives and controls for NN-CDH."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.t1 import records
from model.t1.alternative_adaptation import predict_alternative, retrieved_pairs, make_error_memory
from model.t1.alternative_training import fit_attention, fit_point_forests
from model.t1.pairwise_training import fit_forest
from tools import t1_pairwise_redesign as p

EXPERIMENT = "alt_v1"
FAMILIES = ("direct_trees", "local_pair_trees", "anchor_transport", "error_memory", "vote_attention")
TAGS = dict(zip(FAMILIES, ("d", "l", "t", "m", "a")))
PROTOCOL = dict(version=1, source_study="ar1", core="frozen_source_context",
    prediction_selection="val_error_then_val_final_loss_then_id", test="selected_only",
    attention="bounded_log_weight_adjustment_actual_case_targets_epoch_zero_eligible",
    memory="LOO_retrieval_errors_descriptor_scaled_on_training_only_inverse_distance",
    transport="five_fold_point_predictions_each_anchor_excluded_from_its_head",
    local_pair="retrieved_only_equal_query_mass_both_directions",
    direct="same_frozen_representations_not_raw_features")
FOREST = dict(n_estimators=64, tree_step=16, max_depth=12, max_features=.7)
ATTENTION = dict(hidden=[64, 32], score_cap=4., lr=.001, weight_decay=1e-4,
    grad_clip=5., batch_size=256, epochs=120, patience=20)
CANDIDATES = {
    "direct_trees": {f"d{i}": dict(**FOREST, min_samples_leaf=v) for i, v in enumerate((1, 5, 20))},
    "local_pair_trees": {f"l{i}": dict(**FOREST, min_samples_leaf=v) for i, v in enumerate((5, 20))},
    "anchor_transport": {f"t{i}": dict(**FOREST, min_samples_leaf=v, folds=5) for i, v in enumerate((1, 5, 20))},
    "error_memory": {f"m{i}": dict(neighbors=k, multiplier=b) for i, (k, b) in enumerate(((8, .25), (8, 1.), (32, .25), (32, 1.)))},
    "vote_attention": {f"a{i}": dict(**ATTENTION, dropout=v) for i, v in enumerate((0., .1))},
}


def key_for(study, task, seed, candidate):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1], candidate, f"s{seed}")


def source_context(task, seed):
    unit = tracking.load_unit(p.key_for("ar1", task, seed, "ctx"), config=None, arrays=False)
    if unit is None or unit.config["task"] != task or unit.config["seed"] != seed or unit.config["protocol"] != p.PROTOCOL:
        raise ValueError("completed matching ar1 context is required")
    context = p.numpy_state(unit.load_state("context"))
    train = context["train"]
    if ((train["indices"] == np.arange(len(train["indices"]))[:, None]) & (train["weights"] > 0)).any():
        raise AssertionError("source has positive-weight self retrieval")
    return unit, context


def candidate_config(task, seed, family, cid, context_unit):
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, task_type=task.split("__")[0],
        seed=seed, family=family, candidate=cid, settings=CANDIDATES[family][cid],
        context_unit=context_unit.info.run_id, context_config_identity=tracking.config_identity(context_unit.config))


def fit_alternative(context, family, settings, *, seed, run):
    if family == "vote_attention":
        return fit_attention(context, settings, seed=seed, run=run)
    if family in ("direct_trees", "anchor_transport"):
        return fit_point_forests(context, settings, seed=seed, run=run, transport=family == "anchor_transport")
    if family == "local_pair_trees":
        return fit_forest(context, retrieved_pairs(context), settings, seed=seed, run=run)
    if family == "error_memory":
        resumed = run.load_resume()
        if resumed is not None:
            model = records.unpack(resumed.state["model"])
        else:
            model = make_error_memory(context)
            run.save_resume(dict(model=records.pack(model)), step=1)
        return model, dict(history=[], selected_epoch=0, epochs_run=0)
    raise ValueError(family)


def run_candidate(study, task, seed, family, cid, context_unit, context):
    key = key_for(study, task, seed, cid)
    config = candidate_config(task, seed, family, cid, context_unit)
    unit = tracking.load_unit(key, config=config, arrays=False)
    if unit is not None:
        print(f"[recorded] {task} s{seed} {cid}", flush=True)
        return unit
    started = time.perf_counter()
    with tracking.open_run(key, config=config) as run:
        settings = CANDIDATES[family][cid]
        model, info = fit_alternative(context, family, settings, seed=seed, run=run)
        part = context["val"]
        prediction = predict_alternative(model, context, part, family, settings)
        row = dict(task=task, task_type=context["task_type"], seed=seed, family=family, candidate=cid,
            unit=key.run_id, selected_epoch=info["selected_epoch"], epochs_run=info["epochs_run"],
            elapsed_seconds=time.perf_counter() - started,
            **{"val_" + k: v for k, v in p.metrics(prediction, part["target"], context["task_type"], part["p0"]).items()})
        run.save_state("model", records.pack(model))
        run.save_state("val_predictions", records.storable(dict(final=prediction, p0=part["p0"], target=part["target"])))
        run.log_record("row", row)
        run.log_record("history", info["history"])
        run.log_final({k: v for k, v in row.items() if isinstance(v, (int, float))})
        records.save_rng(run)
    print(f"[candidate] {task} s{seed} {cid}: val={row['val_error']:.6f}", flush=True)
    return tracking.load_unit(key, config=config, arrays=False)


def selected_config(task, seed, family, candidates, chosen):
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, task_type=task.split("__")[0], seed=seed,
        family=family, role="selected", selected_candidate=chosen["candidate"], selected_unit=chosen["unit"],
        candidates=[dict(unit=u.info.run_id, config_identity=tracking.config_identity(u.config)) for u in candidates])


def evaluate_selected(study, task, seed, family, candidates, context):
    rows = [records.record(u, "row") for u in candidates]
    chosen = p.choose_candidate(rows)
    key = key_for(study, task, seed, "e_" + TAGS[family])
    config = selected_config(task, seed, family, candidates, chosen)
    existing = tracking.load_unit(key, config=config, arrays=False)
    if existing is not None:
        print(f"[recorded-eval] {task} s{seed} {family}", flush=True)
        return existing
    selected = next(u for u in candidates if u.info.run_id == chosen["unit"])
    with tracking.open_run(key, config=config) as run:
        run.load_resume()
        model = records.unpack(selected.load_state("model"))
        settings = CANDIDATES[family][chosen["candidate"]]
        restored = predict_alternative(model, context, context["val"], family, settings)
        if not np.array_equal(restored, selected.load_state("val_predictions")["final"].numpy()):
            raise AssertionError("restored validation prediction differs")
        part = context["test"]
        final = predict_alternative(model, context, part, family, settings)
        if not np.array_equal(final, predict_alternative(model, context, part, family, settings)):
            raise AssertionError("non-deterministic test inference")
        row = dict(chosen, method=family, adapter=family, scheme="sequential", unit=key.run_id,
            candidate_unit=chosen["unit"], initial_core_hash=context["core_hash"], selected_core_hash=context["core_hash"],
            test_pre_error=p.metrics(part["p0"], part["target"], context["task_type"])["error"],
            **{"test_" + k: v for k, v in p.metrics(final, part["target"], context["task_type"], part["p0"]).items()})
        run.save_state("test_predictions", records.storable(dict(final=final, p0=part["p0"], target=part["target"])))
        run.log_record("row", row)
        run.log_record("selection", dict(selected=chosen["candidate"], candidates=rows))
        run.log_final({k: v for k, v in row.items() if isinstance(v, (int, float))})
        records.save_rng(run)
    print(f"[selected] {task} s{seed} {family}: test={row['test_error']:.6f}", flush=True)
    return tracking.load_unit(key, config=config, arrays=False)


def run_split(study, task, seed, families=FAMILIES):
    unit, context = source_context(task, seed)
    for family in families:
        candidates = [run_candidate(study, task, seed, family, cid, unit, context) for cid in CANDIDATES[family]]
        evaluate_selected(study, task, seed, family, candidates, context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", choices=p.q.prior.TASKS, default=list(p.q.prior.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(p.q.prior.SEEDS))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args(argv)
    records.check_study(args.study)
    if args.mode == "summarize":
        from tools.t1_alternative_redesign_summary import summarize
        return summarize(args.study, args.tasks, args.seeds)
    for task in args.tasks:
        for seed in args.seeds:
            run_split(args.study, task, seed, args.families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
