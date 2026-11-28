"""Recorded Random Forest comparison with the existing ExtraTrees pair recipes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.t1 import records
from model.t1.alternative_adaptation import retrieved_pairs
from model.t1.pairwise_adaptation import pair_training_data, predict_pairs
from model.t1.pairwise_training import fit_forest
from tools import t1_pairwise_redesign as p
from tools import t1_alternative_redesign as a

EXPERIMENT = "rf_v1"
FAMILIES = ("pair_random_forest", "local_pair_random_forest")
TAGS = dict(zip(FAMILIES, ("m", "l")))
PROTOCOL = dict(version=1, source_study="ar1", reference_studies=["ar1", "ar2"],
    core="frozen_source_context", estimator="sklearn_random_forest_best_splits_bootstrap_true",
    mixed_pairs="identical_ar1_half_retrieved_half_eight_uniform_nonself_equal_query_mass",
    local_pairs="identical_ar2_retrieved_only_equal_query_mass",
    regression="unchanged_antisymmetric_difference_weighted_anchor_average",
    classification="unchanged_symmetric_same_class_probability_prior_redistribution",
    selection="val_error_then_val_final_loss_then_id", test="selected_only",
    policy_order="existing_ar2_order_then_mixed_RF_then_retrieved_RF")
CANDIDATES = {
    family: {f"{TAGS[family]}{i}": dict(settings, forest_kind="random_forest", bootstrap=True)
        for i, settings in enumerate(source.values())}
    for family, source in zip(FAMILIES, (p.CANDIDATES["pair_trees"], a.CANDIDATES["local_pair_trees"]))
}


def key_for(study, task, seed, candidate):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1], candidate, f"s{seed}")


def candidate_config(task, seed, family, cid, context_unit):
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, task_type=task.split("__")[0],
        seed=seed, family=family, candidate=cid, settings=CANDIDATES[family][cid],
        context_unit=context_unit.info.run_id, context_config_identity=tracking.config_identity(context_unit.config))


def training_pairs(context, family, seed):
    if family == "pair_random_forest":
        return pair_training_data(context, seed=seed, random_anchors=p.PROTOCOL["random_anchors"])
    if family == "local_pair_random_forest":
        return retrieved_pairs(context)
    raise ValueError(family)


def pair_digest(pairs):
    digest = hashlib.sha256()
    for name in sorted(pairs):
        value = np.ascontiguousarray(pairs[name])
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def predict(model, context, part):
    return predict_pairs(model, context, part, family="pair_trees")


def run_candidate(study, task, seed, family, cid, context_unit, context):
    key = key_for(study, task, seed, cid)
    config = candidate_config(task, seed, family, cid, context_unit)
    unit = tracking.load_unit(key, config=config, arrays=False)
    if unit is not None:
        print(f"[recorded] {task} s{seed} {cid}", flush=True)
        return unit
    started = time.perf_counter()
    with tracking.open_run(key, config=config) as run:
        pairs = training_pairs(context, family, seed)
        model, info = fit_forest(context, pairs, CANDIDATES[family][cid], seed=seed, run=run)
        part = context["val"]
        prediction = predict(model, context, part)
        row = dict(task=task, task_type=context["task_type"], seed=seed, family=family, candidate=cid,
            unit=key.run_id, selected_epoch=info["selected_epoch"], epochs_run=info["epochs_run"],
            pair_digest=pair_digest(pairs), n_pairs=len(pairs["a"]),
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
        restored = predict(model, context, context["val"])
        if not np.array_equal(restored, selected.load_state("val_predictions")["final"].numpy()):
            raise AssertionError("restored validation prediction differs")
        part = context["test"]
        final = predict(model, context, part)
        if not np.array_equal(final, predict(model, context, part)):
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
    unit, context = a.source_context(task, seed)
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
        return summarize("ar2", args.tasks, args.seeds, forest_study=args.study)
    for task in args.tasks:
        for seed in args.seeds:
            run_split(args.study, task, seed, args.families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
