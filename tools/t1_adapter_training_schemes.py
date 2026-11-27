"""Recorded, objective-matched sequential / alternating / joint adapter study."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from model.common import tracking
from model.common.core import clone_optimizer
from model.t1 import records
from model.t1.calibration import FreeRadius
from model.t1.sync import SyncConfig, train_synchronized
from tools import t1_query_aware as q
from tools import t1_query_no_prediction as n

EXPERIMENT = "schemes_v1"
DESIGNS = {"original": "residual", "query_aware": "query_aware",
           "query_no_prediction": "query_aware_no_prediction"}
SCHEMES = ("sequential", "alternating", "joint")
TAGS = {"original": "o", "query_aware": "q", "query_no_prediction": "n"}
PROTOCOL = dict(version=1, initialization="shared_am1_pretrained_core_and_optimizer",
    sequential="freeze_core", alternating="one_R_then_one_A_pass_shared_permutation",
    objective="match_effective_original_component_objectives_including_summed",
    budget="same_maximum_component_passes_and_original_early_stopping",
    diabetes="reuse_frozen_sequential_extend_separated_LR_pre_plus_post_LA_combined",
    scheme_selection="validation_task_loss_then_error_then_sequential_alternating_joint",
    original_study="am1", query_study="aq1", no_prediction_study="aq2")


def key_for(study, task, seed, adapter, scheme):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1],
                       f"{scheme[:3]}_{TAGS[adapter]}", f"s{seed}")


def references(task, seed):
    refs = q.source_refs(task, seed)
    _, source, core, _, fp = refs
    query = tracking.load_unit(q.key_for("aq1", task, seed, "query_aware"),
        config=q.config_for(task, seed, "query_aware", source, core, fp), arrays=False)
    if query is None:
        raise ValueError("completed aq1 query reference is required")
    nopred = tracking.load_unit(n.key_for("aq2", task, seed),
        config=n.config_for(task, seed, refs, query), arrays=False)
    if nopred is None:
        raise ValueError("completed aq2 reference is required")
    return refs, {"original": source, "query_aware": query, "query_no_prediction": nopred}


def reused_scheme(task):
    return "sequential" if task == "regression__diabets" else "joint"


def settings(description, design):
    if "sync" in description:
        cfg = SyncConfig(**{**description["sync"], "adapter_design": design})
        if cfg.schedule != "joint_rr" or cfg.maintains or not description["core_pretrain"]:
            raise ValueError("expected the recorded pretrained joint condition without maintenance")
        return cfg, description["retrieval_lr_scale"], description["near_scale"], description["free_radius_at_t_star"]
    a = description["adapter"]
    if a["loss"] != "combined":
        raise ValueError("Diabetes mapping expects combined reuse loss")
    cfg = SyncConfig(schedule="joint_rr", w_pre=1., w_post=1., w_near=0.,
        v_post=a["lambda_cls"], v_delta=a["lambda_diff"], v_small=0.,
        output_mode="aggregate_regression", epochs=a["epochs"], patience=a["patience"],
        adapter_lr=a["lr"], hidden_dims=a["hidden_dims"], seed=a["seed"],
        joint_objective="separated", adapter_warmup_epochs=0, diagnostics_every=0,
        adapter_weight_decay=a["weight_decay"], adapter_design=design)
    return cfg, 1., 1., None


def config_for(task, seed, adapter, scheme, refs):
    _, source, core, _, fp = refs
    desc = source.load_state("checkpoint")["model_description"]
    cfg, lr_scale, near_scale, radius = settings(desc, DESIGNS[adapter])
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, seed=seed,
        adapter=adapter, scheme=scheme, sync=asdict(cfg), retrieval_lr_scale=lr_scale,
        near_scale=near_scale, free_radius=radius,
        source_unit=source.info.run_id, source_config_identity=tracking.config_identity(source.config),
        core_unit=core.info.run_id, core_config_identity=tracking.config_identity(core.config),
        data_fingerprint=fp)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def component_counts(history, selected_epoch, batches=None):
    result = dict(epochs_run=len(history), selected_epoch=int(selected_epoch))
    for component in ("retrieval", "adapter"):
        def count(h):
            field = component + "_updates"
            if field in h:
                return h[field]
            phase = h.get("phase", "adapter")
            return h.get("optimizer_updates", batches or 0) if phase in (component, "joint") else 0
        result[component + "_updates_executed"] = sum(count(h) for h in history)
        result[component + "_updates_selected"] = sum(count(h) for h in history if h["epoch"] <= selected_epoch)
    return result


def task_loss(final, target, kind, score_scale=1.):
    if kind == "classification":
        return float(F.cross_entropy(final * score_scale, target.long()))
    return float(F.mse_loss(final, target.reshape_as(final)))


def run_one(study, task, seed, adapter_name, scheme, refs=None, sources=None):
    if refs is None:
        refs, sources = references(task, seed)
    if scheme == reused_scheme(task):
        print(f"[reference] {task} s{seed} {scheme} {adapter_name}", flush=True)
        return
    config = config_for(task, seed, adapter_name, scheme, refs)
    key = key_for(study, task, seed, adapter_name, scheme)
    if tracking.load_unit(key, config=config, arrays=False) is not None:
        print(f"[recorded] {task} s{seed} {scheme} {adapter_name}", flush=True)
        return
    data = n.load_data(task, seed, refs)
    _, source, core, _, _ = refs
    with tracking.open_run(key, config=config) as run:
        base = records.unpack(core.load_state("objects"))
        records.restore_rng_of(core)
        model, cc, tr = base["model"], base["cc"], base["tr"]
        description = copy.deepcopy(source.load_state("checkpoint")["model_description"])
        cfg, lr_scale, near_scale, radius = settings(description, DESIGNS[adapter_name])
        if "sync" not in description:
            cc = replace(cc, batch_size=description["adapter"]["batch_size"])
        optimizer = clone_optimizer(tr.optimizer, model, cc)
        for group in optimizer.param_groups:
            group["lr"] *= lr_scale
        initial_hash = state_hash(model.state_dict())
        initial = {split: q.context(model, getattr(data, "X_" + split))["p0"] for split in ("val", "test")}
        run.save_state("initial_predictions", records.storable(initial))
        adapter, info = train_synchronized(model, data, cc, cfg,
            None if radius is None else FreeRadius(**radius), near_scale=near_scale,
            core_optimizer=optimizer, run=run,
            comparison_scheme=None if scheme == "joint" else scheme)
        final_hash = state_hash(model.state_dict())
        if scheme == "sequential" and initial_hash != final_hash:
            raise AssertionError("sequential retrieval parameters changed")
        description.update(sync=asdict(cfg), comparison_scheme=scheme, core=asdict(cc),
            free_radius_selected=info["free_radius"])
        row = dict(task=task, task_type=data.task_type, seed=seed, adapter=adapter_name, scheme=scheme,
            method=f"{scheme}__{adapter_name}", reused=False, unit=key.run_id,
            initial_core_hash=initial_hash, selected_core_hash=final_hash,
            adapter_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
            elapsed_seconds=info["elapsed_seconds"],
            **component_counts(info["history"], info["best_epoch"]))
        for split in ("val", "test"):
            target = getattr(data, "y_" + split)
            reference = sources["query_aware"].load_state(split + "_predictions")
            if not torch.equal(reference["target"], target):
                raise AssertionError("paired targets differ")
            values = q.context(model, getattr(data, "X_" + split))
            final = q.predict_context(adapter, data.task_type, values)
            if scheme == "sequential" and not torch.equal(values["p0"], initial[split]):
                raise AssertionError("sequential retrieval predictions changed")
            row.update({split + "_" + k: v for k, v in q.metrics(final, target, data.task_type, values["p0"]).items()})
            row[split + "_task_loss"] = task_loss(final, target, data.task_type, cfg.score_scale)
            row[split + "_initial_error"] = q.metrics(initial[split], target, data.task_type)["error"]
            run.save_state(split + "_predictions", records.storable(dict(**values, final=final, target=target)))
        if not np.isclose(row["val_task_loss"], info["best_val_L_post"], atol=2e-6, rtol=1e-5):
            raise AssertionError("selected validation loss differs")
        run.save_state("checkpoint", records.storable(dict(model_state=model.state_dict(),
            adapter_state=adapter.state_dict(), model_description=description, training_info=info)))
        run.log_record("history", records.ordered(info["history"]))
        run.log_records("rows", [records.ordered(row)])
        run.log_final({k: v for k, v in row.items() if isinstance(v, (int, float))})
        records.save_rng(run)
    print(f"[done] {task} s{seed} {scheme} {adapter_name}: error={row['test_error']:.6f}, "
          f"epoch={row['selected_epoch']}/{row['epochs_run']}", flush=True)


def choose_scheme(rows):
    """No test field participates in schedule selection."""
    return min(rows, key=lambda r: (r["val_task_loss"], r["val_error"], SCHEMES.index(r["scheme"])))


def reused_row(task, seed, adapter, unit, query_unit, batches, initial_hash):
    scheme = reused_scheme(task)
    checkpoint = unit.load_state("checkpoint")
    description = checkpoint["model_description"]
    cfg, _, _, _ = settings(description, DESIGNS[adapter])
    history = records.record(unit, "history")
    if adapter == "original":
        old = records.record(unit, "row")
        history = history.get("sync", history.get("adapter"))
        epoch = checkpoint["optimizer_checkpoint_epoch"]["adapter"]
        stored = query_unit.load_state("intervention_predictions")
        saved = dict(p0=stored["original_context"]["p0"], final=stored["original_normal"], target=stored["target"])
        kind = task.split("__")[0]
        row = dict(val_task_loss=old["val_L_post"],
            val_error=1-old["val_accuracy_post"] if kind == "classification" else old["val_rmse_post"],
            **{"test_"+k: v for k, v in q.metrics(saved["final"], saved["target"], kind, saved["p0"]).items()})
        expected = 1-old["test_accuracy_post"] if kind == "classification" else old["test_rmse_post"]
        if not np.isclose(row["test_error"], expected, atol=2e-6, rtol=1e-5):
            raise AssertionError("original reference prediction mismatch")
    else:
        row, = records.rows(unit, "rows")
        row = dict(row)
        epoch = row["selected_epoch"]
        val = unit.load_state("val_predictions")
        row["val_task_loss"] = task_loss(val["final"], val["target"], row["task_type"], cfg.score_scale)
        saved = unit.load_state("test_predictions")
    row.update(task=task, task_type=task.split("__")[0], seed=seed, adapter=adapter, scheme=scheme,
        method=f"{scheme}__{adapter}", reused=True, unit=unit.info.run_id,
        initial_core_hash=initial_hash, selected_core_hash=state_hash(checkpoint["model_state"]),
        **component_counts(history, epoch, batches))
    return row, saved


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, tasks, seeds):
    task_rows = []
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for task in tasks:
        for method in methods:
            part = [r for r in rows if r["task"] == task and r["method"] == method]
            if len(part) != len(seeds):
                raise AssertionError("incomplete task aggregation")
            item = dict(task=task, task_type=part[0]["task_type"], method=method, n_seeds=len(part),
                test_error=float(np.mean([r["test_error"] for r in part])))
            if "adapter" in part[0]:
                item.update(adapter=part[0]["adapter"], scheme=part[0]["scheme"])
            for key in ("test_pre_error", "test_helped", "test_harmed", "test_n"):
                if all(key in r for r in part):
                    item[key] = float(np.mean([r[key] for r in part])) if key.endswith("error") else sum(r[key] for r in part)
            task_rows.append(item)
    families = []
    for kind in ("classification", "regression"):
        for method in methods:
            part = [r for r in task_rows if r["task_type"] == kind and r["method"] == method]
            if not part:
                continue
            item = dict(task_type=kind, method=method, n_tasks=len(part),
                mean_error=float(np.mean([r["test_error"] for r in part])))
            if "adapter" in part[0]:
                item.update(adapter=part[0]["adapter"], scheme=part[0]["scheme"])
            if all("test_pre_error" in r for r in part):
                item["mean_pre_error"] = float(np.mean([r["test_pre_error"] for r in part]))
            for key in ("test_helped", "test_harmed", "test_n"):
                if all(key in r for r in part):
                    item[key] = sum(r[key] for r in part)
            families.append(item)
    return task_rows, families


def summarize(study, tasks, seeds):
    """Read only recorded units. Never train or make fresh model predictions here."""
    from tools.t1_adaptation_diagnostics import regression_diagnostics, classification_diagnostics
    rows, behavior, harms, missing = [], [], [], []
    prediction_checks = frozen_hash_checks = sequential_prediction_checks = 0
    for task in tasks:
        kind = task.split("__")[0]
        for seed in seeds:
            refs, sources = references(task, seed)
            data = n.load_data(task, seed, refs)
            base = records.unpack(refs[2].load_state("objects"))
            initial_hash = state_hash(base["model"].state_dict())
            desc = sources["original"].load_state("checkpoint")["model_description"]
            batch_size = desc.get("adapter", {}).get("batch_size", base["cc"].batch_size)
            batches = (len(data.y_train)+batch_size-1)//batch_size
            del base
            retrieval_row = records.record(refs[3], "row")
            initial_error = 1-retrieval_row["test_accuracy_pre"] if kind == "classification" else retrieval_row["test_rmse_pre"]
            sequential_pre = None
            for adapter in DESIGNS:
                for scheme in SCHEMES:
                    if scheme == reused_scheme(task):
                        unit = sources[adapter]
                        row, saved = reused_row(task, seed, adapter, unit, sources["query_aware"], batches, initial_hash)
                        splits = ("test",) if adapter == "original" else ("val", "test")
                    else:
                        key = key_for(study, task, seed, adapter, scheme)
                        unit = tracking.load_unit(key, config=config_for(task, seed, adapter, scheme, refs), arrays=False)
                        if unit is None:
                            missing.append(key.run_id)
                            continue
                        row, = records.rows(unit, "rows")
                        saved = unit.load_state("test_predictions")
                        splits = ("val", "test")
                        if row["initial_core_hash"] != initial_hash:
                            raise AssertionError("different starting retrieval state")
                        if state_hash(unit.load_state("checkpoint")["model_state"]) != row["selected_core_hash"]:
                            raise AssertionError("selected checkpoint hash mismatch")
                        if not np.isclose(row["test_initial_error"], initial_error, atol=2e-6, rtol=1e-5):
                            raise AssertionError("starting prediction differs from pretrained retrieval baseline")
                    row["test_initial_error"] = initial_error
                    cfg, _, _, _ = settings(desc, DESIGNS[adapter])
                    for split in splits:
                        values = saved if split == "test" else unit.load_state("val_predictions")
                        if not torch.equal(values["target"], getattr(data, "y_"+split)):
                            raise AssertionError("paired target mismatch")
                        if not all(torch.isfinite(values[k]).all() for k in ("p0", "final", "target")):
                            raise AssertionError("nonfinite saved prediction")
                        measured = q.metrics(values["final"], values["target"], kind, values["p0"])
                        for field, value in measured.items():
                            if split+"_"+field in row and not np.isclose(value, row[split+"_"+field], atol=2e-6, rtol=1e-5):
                                raise AssertionError("recorded prediction metric mismatch")
                        if split == "val" and not np.isclose(task_loss(values["final"], values["target"], kind, cfg.score_scale),
                                row["val_task_loss"], atol=2e-6, rtol=1e-5):
                            raise AssertionError("validation selection loss mismatch")
                        prediction_checks += 1
                    if scheme == "sequential":
                        if row["selected_core_hash"] != initial_hash:
                            raise AssertionError("sequential reference core changed")
                        frozen_hash_checks += 1
                        if sequential_pre is not None:
                            if not torch.equal(sequential_pre, saved["p0"]):
                                raise AssertionError("sequential adapters use different retrieval predictions")
                            sequential_prediction_checks += 1
                        sequential_pre = saved["p0"]
                    rows.append(row)
                    identity = {k: row[k] for k in ("task", "task_type", "seed", "adapter", "scheme", "method")}
                    fn = classification_diagnostics if kind == "classification" else regression_diagnostics
                    a, b, y = (saved[k].numpy() for k in ("p0", "final", "target"))
                    if kind == "regression":
                        a, b, y = (v.reshape(-1) for v in (a, b, y))
                    stats, cases, _ = fn(a, b, y)
                    behavior.append(dict(**identity, **{k: v for k, v in stats.items() if not k.startswith("oracle_")}))
                    if kind == "regression":
                        for index, case in sorted(enumerate(cases), key=lambda item: item[1]["squared_error_change"], reverse=True)[:5]:
                            if case["squared_error_change"] > 0:
                                harms.append(dict(**identity, query_index=index, retrieval=float(a[index]),
                                    final=float(b[index]), target=float(y[index]), **case))
            print(f"[read] {task} s{seed}", flush=True)
    if missing:
        raise ValueError(f"{len(missing)} missing units: {missing[:5]}")
    if len(rows) != len(tasks)*len(seeds)*9 or len({(r["task"], r["seed"], r["method"]) for r in rows}) != len(rows):
        raise AssertionError("incomplete or duplicated comparison")
    task_rows, families = aggregate(rows, tasks, seeds)
    selected = []
    for task in tasks:
        for seed in seeds:
            for adapter in DESIGNS:
                chosen = choose_scheme([r for r in rows if r["task"] == task and r["seed"] == seed and r["adapter"] == adapter])
                selected.append({**chosen, "selected_scheme": chosen["scheme"], "scheme": "validation_selected",
                                 "method": f"validation_selected__{adapter}"})
    selected_tasks, selected_families = aggregate(selected, tasks, seeds)
    baseline = q.summarize("aq1", tasks, seeds, write=False)
    if baseline["missing"]:
        raise AssertionError("missing compact baseline reference")
    comparison_tasks = task_rows + selected_tasks + [r for r in baseline["tasks"] if r["method"] in ("retrieval_only", *q.BASELINES)]
    comparison_runs = rows + selected + [r for r in baseline["rows"] if r["method"] in ("retrieval_only", *q.BASELINES)]
    pairs = []
    for kind in ("classification", "regression"):
        for adapter in DESIGNS:
            for scheme in (*SCHEMES, "validation_selected"):
                method = f"{scheme}__{adapter}"
                for reference in [f"{s}__{adapter}" for s in SCHEMES if s != scheme] + ["retrieval_only", *q.BASELINES]:
                    matched = [(a, next(b for b in comparison_tasks if b["task"] == a["task"] and b["method"] == reference))
                               for a in comparison_tasks if a["task_type"] == kind and a["method"] == method]
                    if matched:
                        pairs.append(dict(task_type=kind, method=method, reference=reference, **n.ratio_summary(matched)))
    # Compare adapter inputs within each schedule as well.
    for kind in ("classification", "regression"):
        for scheme in (*SCHEMES, "validation_selected"):
            for adapter in ("query_aware", "query_no_prediction"):
                method, reference = f"{scheme}__{adapter}", f"{scheme}__original"
                matched = [(a, next(b for b in comparison_tasks if b["task"] == a["task"] and b["method"] == reference))
                           for a in comparison_tasks if a["task_type"] == kind and a["method"] == method]
                if matched:
                    pairs.append(dict(task_type=kind, method=method, reference=reference, **n.ratio_summary(matched)))
    for comparison in pairs:
        left = [r for r in comparison_runs if r["method"] == comparison["method"] and r["task_type"] == comparison["task_type"]]
        differences = []
        for a in left:
            b = next(r for r in comparison_runs if r["method"] == comparison["reference"] and r["task"] == a["task"] and r["seed"] == a["seed"])
            differences.append(a["test_error"]-b["test_error"])
        comparison.update(seed_pairs=len(differences), seed_wins=sum(d < -1e-7 for d in differences),
            seed_ties=sum(abs(d) <= 1e-7 for d in differences), seed_losses=sum(d > 1e-7 for d in differences))
    selection_counts = [dict(task_type=kind, adapter=adapter,
        **{s: sum(r["selected_scheme"] == s for r in selected if r["task_type"] == kind and r["adapter"] == adapter) for s in SCHEMES})
        for kind in ("classification", "regression") for adapter in DESIGNS
        if any(r["task_type"] == kind for r in selected)]
    behavior_totals, budgets = [], []
    for family in families:
        kind, method = family["task_type"], family["method"]
        part = [r for r in behavior if r["task_type"] == kind and r["method"] == method]
        totals = {k: family[k] for k in ("task_type", "method", "adapter", "scheme")}
        for k in part[0]:
            if k.startswith("count_") or k in {"n", "improved", "worsened", "pre_wrong", "post_wrong", "margin_improves", "margin_worsens"} or k.endswith("_sse"):
                totals[k] = sum(r[k] for r in part)
        behavior_totals.append(totals)
        part = [r for r in rows if r["task_type"] == kind and r["method"] == method]
        budget = {k: family[k] for k in ("task_type", "method", "adapter", "scheme")}
        for field in ("epochs_run", "selected_epoch", "retrieval_updates_executed", "retrieval_updates_selected", "adapter_updates_executed", "adapter_updates_selected"):
            budget["mean_"+field] = float(np.mean([r[field] for r in part]))
            budget["total_"+field] = sum(r[field] for r in part)
        budget["zero_selected_retrieval_updates"] = sum(r["retrieval_updates_selected"] == 0 for r in part)
        budgets.append(budget)
    report = dict(study=study, protocol=PROTOCOL, tasks_included=list(tasks), seeds=list(seeds),
        rows=rows, tasks=task_rows, families=families, comparisons=pairs, selected=selected,
        selected_tasks=selected_tasks, selected_families=selected_families, selection_counts=selection_counts, behavior=behavior,
        behavior_totals=behavior_totals, harm_examples=harms, budgets=budgets,
        previous_mixed_schedule=baseline["families"], baseline_tasks=baseline["tasks"],
        prediction_checks=prediction_checks, baseline_prediction_checks=baseline["prediction_checks"],
        frozen_hash_checks=frozen_hash_checks, sequential_prediction_checks=sequential_prediction_checks,
        new_units=sum(not r["reused"] for r in rows), reused_units=sum(r["reused"] for r in rows))
    out = REPO / "results" / "adaptation_improvement" / study
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for name in ("rows", "tasks", "families", "comparisons", "selected", "selected_tasks", "selected_families", "selection_counts", "behavior", "behavior_totals", "harm_examples", "budgets"):
        write_csv(out / (name+".csv"), report[name])
    plot_summary(report, out)
    print(json.dumps(dict(rows=len(rows), new=report["new_units"], reused=report["reused_units"],
        prediction_checks=prediction_checks, frozen_hash_checks=frozen_hash_checks, summary=str(out / "comparison.json"))), flush=True)
    return report


def plot_summary(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Original", "Query + prediction", "Query, no prediction"]
    kinds = [k for k in ("classification", "regression") if any(r["task_type"] == k for r in report["families"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5*len(kinds), 4.1), layout="constrained")
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        values = np.array([[next(r["mean_error"] for r in report["families"]
            if r["task_type"] == kind and r["adapter"] == a and r["scheme"] == s) for s in SCHEMES] for a in DESIGNS])
        if kind == "classification":
            values *= 100
        ax.imshow(values, cmap="YlOrRd", aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=12,
                    color="white" if values[i,j] > (values.max()+values.min())/2 else "black")
        ax.set_xticks(range(3), [s.capitalize() for s in SCHEMES])
        ax.set_yticks(range(3), labels)
        ax.set_title("Classification error (%)" if kind == "classification" else "Regression RMSE (standardized)")
    fig.suptitle("Training schemes: equal task means, three seeds; lower is better")
    fig.savefig(out / "scheme_means.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=q.prior.TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=q.prior.SEEDS)
    parser.add_argument("--adapters", nargs="+", choices=tuple(DESIGNS), default=list(DESIGNS))
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    args = parser.parse_args(argv)
    records.check_study(args.study)
    if args.mode == "summarize":
        return summarize(args.study, args.tasks, args.seeds)
    torch.set_num_threads(1)
    records.check_unit_paths(args.study, [(EXPERIMENT, t.split("__", 1)[1], f"{s[:3]}_{TAGS[a]}", f"s{seed}")
        for t in args.tasks for seed in args.seeds for a in args.adapters for s in args.schemes])
    for task in args.tasks:
        for seed in args.seeds:
            refs, sources = references(task, seed)
            for adapter in args.adapters:
                for scheme in args.schemes:
                    run_one(args.study, task, seed, adapter, scheme, refs, sources)


if __name__ == "__main__":
    main()
