"""Recorded alternating retrieval/forest continuations of four sequential winners."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.common.core import clone_optimizer
from model.t1 import records
from model.t1.forest_alternating import extract_context, train_retrieval_block, neighborhood_drift
from model.t1.pairwise_adaptation import pair_training_data, predict_pairs, metrics
from model.t1.pairwise_training import fit_forest
from model.t1.alternative_adaptation import retrieved_pairs
from tools import t1_pairwise_redesign as p
from tools import t1_alternative_redesign as a
from tools import t1_pair_random_forest as rf

EXPERIMENT = "af_v1"
FAMILIES = ("pair_trees", "local_pair_trees", "pair_random_forest", "local_pair_random_forest")
TAGS = dict(zip(FAMILIES, ("em", "el", "rm", "rl")))
PROTOCOL = dict(version=1, cycles=5, retrieval_epochs_per_cycle=2, retrieval_lr_scale=.1,
    objective="L_retrieval_plus_L_final_plus_original_case_bias_penalty",
    retrieval="continue_pretrained_Adam_all_original_trainable_parameters",
    gradient="detach_forest_outputs_keep_activation_and_p0_paths",
    forest="refit_64_trees_same_seed_and_sequential_selected_leaf",
    normalization="fixed_within_retrieval_block_refresh_training_only_before_refit",
    random_anchors="same_seed_and_ids_each_cycle", cases="unchanged_no_maintenance",
    selection="val_error_then_val_loss_then_earlier_cycle_including_zero", test="selected_only",
    sources=["ar1", "ar2", "ar3"], baseline="recorded_sequential_family_winner")


def key_for(study, task, seed, step):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1], step, f"s{seed}")


def identity(unit):
    return dict(unit=unit.info.run_id, config_identity=tracking.config_identity(unit.config))


def source_family(task, seed, family, context_unit):
    driver, study = (p, "ar1") if family == "pair_trees" else (a, "ar2") if family == "local_pair_trees" else (rf, "ar3")
    step = "eval_trees" if driver is p else "e_" + driver.TAGS[family]
    selected = tracking.load_unit(driver.key_for(study, task, seed, step), config=None, arrays=False)
    if selected is None or selected.config["protocol"] != driver.PROTOCOL or selected.config["family"] != family:
        raise ValueError("matching completed sequential winner required")
    cid = selected.config["selected_candidate"]
    candidate = tracking.load_unit(driver.key_for(study, task, seed, cid),
        config=driver.candidate_config(task, seed, family, cid, context_unit), arrays=False)
    if candidate is None:
        raise ValueError("sequential winning forest is missing")
    settings = dict(driver.CANDIDATES[family][cid])
    settings.setdefault("forest_kind", "extra_trees")
    settings.setdefault("bootstrap", False)
    return dict(selected=selected, candidate=candidate, settings=settings)


def config_for(task, seed, family, cycle, role, source, **dependencies):
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, task_type=task.split("__")[0],
        seed=seed, family=family, cycle=cycle, role=role, settings=source["settings"],
        sequential_reference=identity(source["selected"]),
        **{name: identity(unit) for name, unit in dependencies.items()})


def training_pairs(context, family, seed):
    return retrieved_pairs(context) if family.startswith("local_") else pair_training_data(context, seed=seed, random_anchors=8)


def run_retrieval(study, task, seed, family, cycle, source, core_unit, forest_unit, context_unit, data_loader):
    config = config_for(task, seed, family, cycle, "retrieval", source,
        core_source=core_unit, forest_source=forest_unit, context_source=context_unit)
    key = key_for(study, task, seed, f"r{TAGS[family]}{cycle}")
    existing = tracking.load_unit(key, config=config, arrays=False)
    if existing is not None:
        return existing
    started = time.perf_counter()
    with tracking.open_run(key, config=config) as run:
        objects = records.unpack(core_unit.load_state("objects"))
        model, cc = objects["model"], objects["cc"]
        if "optimizer" in objects:
            optimizer = objects["optimizer"]
        else:
            optimizer = clone_optimizer(objects["tr"].optimizer, model, cc)
            for group in optimizer.param_groups:
                group["lr"] *= PROTOCOL["retrieval_lr_scale"]
        records.restore_rng_of(core_unit)
        context = p.numpy_state(context_unit.load_state("context"))
        if p.s.state_hash(model.state_dict()) != context["core_hash"]:
            raise AssertionError("retrieval source state/context mismatch")
        forest = records.unpack(forest_unit.load_state("model"))
        data = data_loader()
        history = train_retrieval_block(model, optimizer, data, forest, context, cc,
            epochs=PROTOCOL["retrieval_epochs_per_cycle"], seed=seed + 1000 * cycle, run=run)
        updated = extract_context(model, data)
        updated["core_hash"] = p.s.state_hash(model.state_dict())
        updated["data_fingerprint"] = context["data_fingerprint"]
        row = dict(task=task, task_type=data.task_type, seed=seed, family=family, cycle=cycle,
            unit=key.run_id, core_hash=updated["core_hash"], previous_core_hash=context["core_hash"],
            retrieval_updates=sum(r["retrieval_updates"] for r in history),
            **neighborhood_drift(context["train"], updated["train"]),
            elapsed_seconds=time.perf_counter() - started)
        run.save_state("objects", records.pack(dict(model=model, optimizer=optimizer, cc=cc)))
        run.save_state("core_state", records.storable(model.state_dict()))
        run.save_state("context", records.storable(updated))
        run.log_record("row", row)
        run.log_record("history", history)
        run.log_final(records.numeric(row))
        records.save_rng(run)
    print(f"[retrieval] {task} s{seed} {family} cycle={cycle}", flush=True)
    return tracking.load_unit(key, config=config, arrays=False)


def run_refit(study, task, seed, family, cycle, source, retrieval_unit):
    config = config_for(task, seed, family, cycle, "refit", source, retrieval_source=retrieval_unit)
    key = key_for(study, task, seed, f"f{TAGS[family]}{cycle}")
    existing = tracking.load_unit(key, config=config, arrays=False)
    if existing is not None:
        return existing
    started = time.perf_counter()
    with tracking.open_run(key, config=config) as run:
        context = p.numpy_state(retrieval_unit.load_state("context"))
        pairs = training_pairs(context, family, seed)
        forest, info = fit_forest(context, pairs, source["settings"], seed=seed, run=run)
        part = context["val"]
        final = predict_pairs(forest, context, part, family="pair_trees")
        row = dict(task=task, task_type=context["task_type"], seed=seed, family=family,
            cycle=cycle, unit=key.run_id, core_hash=context["core_hash"],
            pair_digest=rf.pair_digest(pairs), n_pairs=len(pairs["a"]),
            elapsed_seconds=time.perf_counter() - started,
            val_pre_error=metrics(part["p0"], part["target"], context["task_type"])["error"],
            **{"val_" + k: v for k, v in metrics(final, part["target"], context["task_type"], part["p0"]).items()})
        run.save_state("model", records.pack(forest))
        run.save_state("val_predictions", records.storable(dict(final=final, p0=part["p0"], target=part["target"])))
        run.log_record("row", row)
        run.log_record("history", info["history"])
        run.log_final(records.numeric(row))
        records.save_rng(run)
    print(f"[refit] {task} s{seed} {family} cycle={cycle}: val={row['val_error']:.6f}", flush=True)
    return tracking.load_unit(key, config=config, arrays=False)


def checkpoint_rows(source, refits):
    baseline = records.record(source["candidate"], "row")
    return [dict(baseline, cycle=0)] + [records.record(unit, "row") for unit in refits]


def choose_checkpoint(rows):
    return min(rows, key=lambda r: (r["val_error"], r["val_task_loss"], r["cycle"]))


def selected_config(task, seed, family, source, retrievals, refits):
    chosen = choose_checkpoint(checkpoint_rows(source, refits))
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, task_type=task.split("__")[0],
        seed=seed, family=family, role="selected", selected_cycle=chosen["cycle"], settings=source["settings"],
        sequential_reference=identity(source["selected"]), baseline_forest=identity(source["candidate"]),
        retrievals=[identity(u) for u in retrievals], refits=[identity(u) for u in refits])


def evaluate_selected(study, task, seed, family, source, retrievals, refits, initial_context, data_loader):
    config = selected_config(task, seed, family, source, retrievals, refits)
    key = key_for(study, task, seed, "e" + TAGS[family])
    existing = tracking.load_unit(key, config=config, arrays=False)
    if existing is not None:
        return existing
    cycle = config["selected_cycle"]
    chosen = choose_checkpoint(checkpoint_rows(source, refits))
    with tracking.open_run(key, config=config) as run:
        run.load_resume()
        if cycle == 0:
            saved = p.numpy_state(source["selected"].load_state("test_predictions"))
            part = initial_context["test"]
            core_hash = initial_context["core_hash"]
        else:
            retrieval, refit = retrievals[cycle - 1], refits[cycle - 1]
            objects = records.unpack(retrieval.load_state("objects"))
            context = p.numpy_state(retrieval.load_state("context"))
            core_hash = p.s.state_hash(objects["model"].state_dict())
            if core_hash != context["core_hash"]:
                raise AssertionError("selected retrieval state differs from forest context")
            forest = records.unpack(refit.load_state("model"))
            val = predict_pairs(forest, context, context["val"], family="pair_trees")
            np.testing.assert_array_equal(val, p.numpy_state(refit.load_state("val_predictions"))["final"])
            test_context = extract_context(objects["model"], data_loader(), splits=("test",), normalizer=context)
            np.testing.assert_array_equal(test_context["case_z"], context["case_z"])
            part = test_context["test"]
            final = predict_pairs(forest, test_context, part, family="pair_trees")
            saved = dict(final=final, p0=part["p0"], target=part["target"])
        np.testing.assert_array_equal(saved["target"], initial_context["test"]["target"])
        row = dict(task=task, task_type=initial_context["task_type"], seed=seed, family=family,
            method=family + "_alternating", adapter=family, scheme="alternating", unit=key.run_id,
            selected_epoch=cycle, selected_cycle=cycle, epochs_run=PROTOCOL["cycles"] * PROTOCOL["retrieval_epochs_per_cycle"],
            candidate=f"cycle_{cycle}", reused_baseline=cycle == 0,
            initial_core_hash=initial_context["core_hash"], selected_core_hash=core_hash,
            retrieval_updates_executed=sum(records.record(u, "row")["retrieval_updates"] for u in retrievals),
            retrieval_updates_selected=sum(records.record(u, "row")["retrieval_updates"] for u in retrievals[:cycle]),
            test_initial_error=metrics(initial_context["test"]["p0"], saved["target"], initial_context["task_type"])["error"],
            test_pre_error=metrics(saved["p0"], saved["target"], initial_context["task_type"])["error"],
            **neighborhood_drift(initial_context["test"], part),
            **{k: v for k, v in chosen.items() if k.startswith("val_")},
            **{"test_" + k: v for k, v in metrics(saved["final"], saved["target"], initial_context["task_type"], saved["p0"]).items()})
        run.save_state("test_predictions", records.storable(saved))
        run.save_state("test_context", records.storable(part))
        run.log_record("row", row)
        run.log_record("selection", dict(selected_cycle=cycle, candidates=checkpoint_rows(source, refits)))
        run.log_final(records.numeric(row))
        records.save_rng(run)
    print(f"[selected] {task} s{seed} {family} cycle={cycle}: test={row['test_error']:.6f}", flush=True)
    return tracking.load_unit(key, config=config, arrays=False)


def run_split(study, task, seed, families=FAMILIES):
    context_unit, context = a.source_context(task, seed)
    refs, _ = p.s.references(task, seed)
    loaded = []
    def data_loader():
        if not loaded:
            loaded.append(p.s.n.load_data(task, seed, refs))
        return loaded[0]
    for family in families:
        source = source_family(task, seed, family, context_unit)
        core, forest, ctx = refs[2], source["candidate"], context_unit
        retrievals, refits = [], []
        for cycle in range(1, PROTOCOL["cycles"] + 1):
            retrieval = run_retrieval(study, task, seed, family, cycle, source, core, forest, ctx, data_loader)
            forest = run_refit(study, task, seed, family, cycle, source, retrieval)
            retrievals.append(retrieval)
            refits.append(forest)
            core, ctx = retrieval, retrieval
        evaluate_selected(study, task, seed, family, source, retrievals, refits, context, data_loader)


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
        return summarize("ar2", args.tasks, args.seeds, forest_study="ar3", alternating_study=args.study)
    for task in args.tasks:
        for seed in args.seeds:
            run_split(args.study, task, seed, args.families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
