"""Recorded hidden-dropout ablation across residual inputs and training schemes."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from model.common import tracking
from model.common.core import clone_optimizer
from model.t1 import records
from model.t1.adapter_regularization import dropout_description
from model.t1.calibration import FreeRadius
from model.t1.reuse import ReuseConfig, train_regression_adapter
from model.t1.sync import train_synchronized
from tools import t1_adapter_training_schemes as s

q, n = s.q, s.n
EXPERIMENT = "drop_v1"
RATES = (0., .1, .2)
PROTOCOL = dict(version=1, rates=list(RATES), dropout_placement="all_hidden",
    dropout_behavior="train_only_independent_masks_eval_off",
    initialization="same_parameters_and_RNG_as_zero_dropout",
    zero_reference="as1_with_am1_aq1_aq2_references",
    training="same_as_as1_including_original_frozen_Diabetes_reuse",
    rate_selection="val_task_loss_then_val_error_then_smaller_rate",
    combined_selection="val_task_loss_then_val_error_then_smaller_rate_then_seq_alt_joint")


def rate_tag(rate):
    if rate not in RATES:
        raise ValueError(f"rate must be one of {RATES}")
    return f"p{round(100*rate):02d}"


def method(adapter, scheme, rate):
    return f"{scheme}__{adapter}__{rate_tag(rate)}"


def key_for(study, task, seed, adapter, scheme, rate):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1],
        f"{scheme[:3]}_{s.TAGS[adapter]}_{rate_tag(rate)}", f"s{seed}")


def zero_unit(task, seed, adapter, scheme, refs, sources):
    if scheme == s.reused_scheme(task):
        return sources[adapter]
    unit = tracking.load_unit(s.key_for("as1", task, seed, adapter, scheme),
        config=s.config_for(task, seed, adapter, scheme, refs), arrays=False)
    if unit is None:
        raise ValueError("the complete matched zero-dropout as1 reference is required")
    return unit


def settings(task, scheme, adapter, rate, description):
    if task == "regression__diabets" and scheme == "sequential":
        cfg = ReuseConfig(**{**description["adapter"], "adapter_design":s.DESIGNS[adapter], "dropout":rate})
        return "reuse", cfg, 1., 1., None
    cfg, lr_scale, near_scale, radius = s.settings(description, s.DESIGNS[adapter])
    return "sync", replace(cfg, adapter_dropout=rate), lr_scale, near_scale, radius


def config_for(task, seed, adapter, scheme, rate, refs, zero):
    _, source, core, _, fp = refs
    desc = source.load_state("checkpoint")["model_description"]
    engine, cfg, lr_scale, near_scale, radius = settings(task, scheme, adapter, rate, desc)
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, task=task, seed=seed, adapter=adapter,
        scheme=scheme, dropout=rate, dropout_placement="all_hidden", engine=engine,
        training_config=asdict(cfg), retrieval_lr_scale=lr_scale, near_scale=near_scale, free_radius=radius,
        source_unit=source.info.run_id, source_config_identity=tracking.config_identity(source.config),
        core_unit=core.info.run_id, core_config_identity=tracking.config_identity(core.config),
        zero_unit=zero.info.run_id, zero_config_identity=tracking.config_identity(zero.config), data_fingerprint=fp)


def active_parameter_count(checkpoint, kind):
    state = checkpoint["adapter_state"]
    return sum(v.numel() for k, v in state.items() if kind == "classification" or k.startswith("adapt_net_final."))


def run_one(study, task, seed, adapter_name, scheme, rate, refs=None, sources=None):
    if refs is None:
        refs, sources = s.references(task, seed)
    zero = zero_unit(task, seed, adapter_name, scheme, refs, sources)
    if rate == 0:
        print(f"[reference] {task} s{seed} {scheme} {adapter_name} p00", flush=True)
        return
    config = config_for(task, seed, adapter_name, scheme, rate, refs, zero)
    key = key_for(study, task, seed, adapter_name, scheme, rate)
    if tracking.load_unit(key, config=config, arrays=False) is not None:
        print(f"[recorded] {task} s{seed} {scheme} {adapter_name} {rate_tag(rate)}", flush=True)
        return
    data = n.load_data(task, seed, refs)
    _, source, core, _, _ = refs
    with tracking.open_run(key, config=config) as run:
        base = records.unpack(core.load_state("objects"))
        records.restore_rng_of(core)
        model, cc, tr = base["model"], base["cc"], base["tr"]
        desc = copy.deepcopy(source.load_state("checkpoint")["model_description"])
        engine, cfg, lr_scale, near_scale, radius = settings(task, scheme, adapter_name, rate, desc)
        initial_hash = s.state_hash(model.state_dict())
        initial = {split:q.context(model, getattr(data, "X_"+split))["p0"] for split in ("val", "test")}
        run.save_state("initial_predictions", records.storable(initial))
        started = time.perf_counter()
        if engine == "reuse":
            adapter, info = train_regression_adapter(model, data, cfg, run=run, dropout_placement="all_hidden")
            desc["adapter"] = asdict(cfg)
            batches = (len(data.y_train)+cfg.batch_size-1)//cfg.batch_size
        else:
            if "sync" not in desc:
                cc = replace(cc, batch_size=desc["adapter"]["batch_size"])
            opt = clone_optimizer(tr.optimizer, model, cc)
            for group in opt.param_groups:
                group["lr"] *= lr_scale
            adapter, info = train_synchronized(model, data, cc, cfg,
                None if radius is None else FreeRadius(**radius), near_scale=near_scale,
                core_optimizer=opt, run=run, comparison_scheme=None if scheme == "joint" else scheme,
                dropout_placement="all_hidden")
            desc.update(sync=asdict(cfg), core=asdict(cc), free_radius_selected=info["free_radius"])
            batches = (len(data.y_train)+cc.batch_size-1)//cc.batch_size
        elapsed = time.perf_counter()-started
        layers = dropout_description(adapter)
        if len(layers) != 2 or any(x["p"] != rate for x in layers):
            raise AssertionError("expected two hidden dropout layers at the declared rate")
        parameter_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        if parameter_count != active_parameter_count(zero.load_state("checkpoint"), data.task_type):
            raise AssertionError("dropout changed trainable parameter count")
        final_hash = s.state_hash(model.state_dict())
        if scheme == "sequential" and final_hash != initial_hash:
            raise AssertionError("sequential retrieval changed")
        desc.update(comparison_scheme=scheme, dropout_placement="all_hidden", dropout_rate=rate,
            dropout_modules=layers, training_engine=engine)
        row = dict(task=task, task_type=data.task_type, seed=seed, adapter=adapter_name, scheme=scheme,
            dropout=rate, rate_setting=rate_tag(rate), method=method(adapter_name, scheme, rate),
            reused=False, unit=key.run_id, zero_unit=zero.info.run_id, dropout_layers=len(layers),
            adapter_parameters=parameter_count, initial_core_hash=initial_hash, selected_core_hash=final_hash,
            elapsed_seconds=elapsed, **s.component_counts(info["history"], info["best_epoch"], batches))
        for split in ("val", "test"):
            target = getattr(data, "y_"+split)
            old_target = sources["query_aware"].load_state(split+"_predictions")["target"]
            if not torch.equal(target, old_target):
                raise AssertionError("paired targets differ")
            values = q.context(model, getattr(data, "X_"+split))
            final = q.predict_context(adapter, data.task_type, values)
            if not torch.equal(final, q.predict_context(adapter, data.task_type, values)):
                raise AssertionError("dropout evaluation is not deterministic")
            if scheme == "sequential" and not torch.equal(values["p0"], initial[split]):
                raise AssertionError("sequential predictions changed")
            row.update({split+"_"+k:v for k,v in q.metrics(final, target, data.task_type, values["p0"]).items()})
            row[split+"_task_loss"] = s.task_loss(final, target, data.task_type, cfg.score_scale)
            row[split+"_initial_error"] = q.metrics(initial[split], target, data.task_type)["error"]
            run.save_state(split+"_predictions", records.storable(dict(**values, final=final, target=target)))
        expected_val = info["best_val_L_post"] if engine == "sync" else next(h["val_l_cls"] for h in info["history"] if h["epoch"] == info["best_epoch"])
        if not np.isclose(row["val_task_loss"], expected_val, atol=2e-6, rtol=1e-5):
            raise AssertionError("selected validation task loss differs")
        run.save_state("checkpoint", records.storable(dict(model_state=model.state_dict(),
            adapter_state=adapter.state_dict(), model_description=desc, training_info=info)))
        run.log_record("history", records.ordered(info["history"]))
        run.log_records("rows", [records.ordered(row)])
        run.log_final({k:v for k,v in row.items() if isinstance(v, (int, float))})
        records.save_rng(run)
    print(f"[done] {task} s{seed} {scheme} {adapter_name} {rate_tag(rate)}: "
          f"error={row['test_error']:.6f}, epoch={row['selected_epoch']}/{row['epochs_run']}", flush=True)


def choose(rows):
    return min(rows, key=lambda r:(r["val_task_loss"], r["val_error"], r["dropout"], s.SCHEMES.index(r["scheme"])))


def aggregate(rows, tasks, seeds):
    task_rows, families = s.aggregate(rows, tasks, seeds)
    for item in task_rows+families:
        example = next(r for r in rows if r["method"] == item["method"])
        item["rate_setting"] = example["rate_setting"]
    return task_rows, families


def policy_row(chosen, policy):
    adapter, scheme = chosen["adapter"], chosen["scheme"]
    row = dict(chosen, selected_dropout=chosen["dropout"], selected_scheme=scheme)
    if policy == "rate":
        row.update(method=f"{scheme}__{adapter}__val_rate", rate_setting="validation_selected")
    elif policy == "both":
        row.update(method=f"val_both__{adapter}", scheme="validation_selected", rate_setting="validation_selected")
    elif policy == "zero_scheme":
        row.update(method=f"val_scheme__{adapter}__p00", scheme="validation_selected", rate_setting="p00")
    else:
        raise ValueError(policy)
    return row


def pair_summary(task_rows, run_rows, candidate, reference, kind):
    candidates = [r for r in task_rows if r["method"] == candidate and r["task_type"] == kind]
    pairs = [(a, next(b for b in task_rows if b["task"] == a["task"] and b["method"] == reference)) for a in candidates]
    if not pairs:
        return None
    changes = []
    for a in run_rows:
        if a["method"] != candidate or a["task_type"] != kind:
            continue
        b = next(b for b in run_rows if b["method"] == reference and b["task"] == a["task"] and b["seed"] == a["seed"])
        changes.append(a["test_error"]-b["test_error"])
    return dict(task_type=kind, method=candidate, reference=reference, **n.ratio_summary(pairs),
        seed_pairs=len(changes), seed_wins=sum(d < -1e-7 for d in changes),
        seed_ties=sum(abs(d) <= 1e-7 for d in changes), seed_losses=sum(d > 1e-7 for d in changes))


def summarize(study, tasks, seeds):
    """Validate recorded predictions and select only with validation fields."""
    from tools.t1_adaptation_diagnostics import classification_diagnostics, regression_diagnostics
    rows, behavior, harms, missing = [], [], [], []
    checks = dict(prediction_sets=0, initial_prediction_sets=0, frozen_core_hashes=0,
        cross_sequential_predictions=0, parameter_counts=0, dropout_metadata=0)
    for task in tasks:
        kind = task.split("__")[0]
        for seed in seeds:
            refs, sources = s.references(task, seed)
            data = n.load_data(task, seed, refs)
            base = records.unpack(refs[2].load_state("objects"))
            initial_hash = s.state_hash(base["model"].state_dict())
            desc = sources["original"].load_state("checkpoint")["model_description"]
            batch_size = desc.get("adapter", {}).get("batch_size", base["cc"].batch_size)
            batches = (len(data.y_train)+batch_size-1)//batch_size
            del base
            frozen_ref = zero_unit(task, seed, "query_aware", "sequential", refs, sources)
            starting = {split:frozen_ref.load_state(split+"_predictions")["p0"] for split in ("val", "test")}
            retrieval = records.record(refs[3], "row")
            initial_error = 1-retrieval["test_accuracy_pre"] if kind == "classification" else retrieval["test_rmse_pre"]
            sequential_pre = None
            for adapter in s.DESIGNS:
                for scheme in s.SCHEMES:
                    zero = zero_unit(task, seed, adapter, scheme, refs, sources)
                    zero_checkpoint = zero.load_state("checkpoint")
                    expected_parameters = active_parameter_count(zero_checkpoint, kind)
                    zero_shapes = {k:tuple(v.shape) for k,v in zero_checkpoint["adapter_state"].items()}
                    for rate in RATES:
                        if rate == 0:
                            unit = zero
                            if scheme == s.reused_scheme(task):
                                row, saved = s.reused_row(task, seed, adapter, zero, sources["query_aware"], batches, initial_hash)
                                splits = ("test",) if adapter == "original" else ("val", "test")
                            else:
                                row, = records.rows(zero, "rows")
                                saved = zero.load_state("test_predictions")
                                splits = ("val", "test")
                            row = dict(row, dropout=0., rate_setting="p00", method=method(adapter,scheme,0.),
                                adapter_parameters=expected_parameters, dropout_layers=0, reused=True)
                            checkpoint = zero_checkpoint
                        else:
                            key = key_for(study, task, seed, adapter, scheme, rate)
                            unit = tracking.load_unit(key, config=config_for(task,seed,adapter,scheme,rate,refs,zero), arrays=False)
                            if unit is None:
                                missing.append(key.run_id)
                                continue
                            row, = records.rows(unit, "rows")
                            saved = unit.load_state("test_predictions")
                            splits = ("val", "test")
                            checkpoint = unit.load_state("checkpoint")
                            modules = checkpoint["model_description"]["dropout_modules"]
                            if row["dropout_layers"] != 2 or len(modules) != 2 or any(m["p"] != rate for m in modules):
                                raise AssertionError("recorded dropout architecture differs")
                            checks["dropout_metadata"] += 1
                            if row["adapter_parameters"] != expected_parameters:
                                raise AssertionError("dropout parameter count differs from zero reference")
                            shapes = {k:tuple(v.shape) for k,v in checkpoint["adapter_state"].items()}
                            if shapes != zero_shapes:
                                raise AssertionError("dropout parameter keys/shapes differ from zero reference")
                            checks["parameter_counts"] += 1
                            initial = unit.load_state("initial_predictions")
                            for split in ("val", "test"):
                                if not torch.equal(initial[split], starting[split]):
                                    raise AssertionError("initial predictions differ from common frozen reference")
                                checks["initial_prediction_sets"] += 1
                        if row["initial_core_hash"] != initial_hash:
                            raise AssertionError("different starting retrieval state")
                        if s.state_hash(checkpoint["model_state"]) != row["selected_core_hash"]:
                            raise AssertionError("checkpoint hash differs from row")
                        if "test_initial_error" in row and not np.isclose(row["test_initial_error"],initial_error,atol=2e-6,rtol=1e-5):
                            raise AssertionError("initial retrieval error differs")
                        row["test_initial_error"] = initial_error
                        _, cfg, _, _, _ = settings(task, scheme, adapter, rate, desc)
                        for split in splits:
                            values = saved if split == "test" else unit.load_state("val_predictions")
                            if not torch.equal(values["target"], getattr(data,"y_"+split)):
                                raise AssertionError("paired target mismatch")
                            if not all(torch.isfinite(values[k]).all() for k in ("p0","final","target")):
                                raise AssertionError("nonfinite prediction")
                            actual = q.metrics(values["final"],values["target"],kind,values["p0"])
                            for field,value in actual.items():
                                if split+"_"+field in row and not np.isclose(value,row[split+"_"+field],atol=2e-6,rtol=1e-5):
                                    raise AssertionError("saved prediction metric mismatch")
                            if split == "val" and not np.isclose(s.task_loss(values["final"],values["target"],kind,cfg.score_scale),row["val_task_loss"],atol=2e-6,rtol=1e-5):
                                raise AssertionError("validation selection loss mismatch")
                            checks["prediction_sets"] += 1
                        if scheme == "sequential":
                            if row["selected_core_hash"] != initial_hash:
                                raise AssertionError("frozen core changed")
                            checks["frozen_core_hashes"] += 1
                            if sequential_pre is not None:
                                if not torch.equal(sequential_pre,saved["p0"]):
                                    raise AssertionError("sequential predictions differ across adapters/rates")
                                checks["cross_sequential_predictions"] += 1
                            sequential_pre = saved["p0"]
                        rows.append(row)
                        identity = {k:row[k] for k in ("task","task_type","seed","adapter","scheme","dropout","rate_setting","method","unit")}
                        a,b,y = (saved[k].numpy() for k in ("p0","final","target"))
                        if kind == "regression":
                            a,b,y = (v.reshape(-1) for v in (a,b,y))
                        fn = classification_diagnostics if kind == "classification" else regression_diagnostics
                        stats,cases,_ = fn(a,b,y)
                        behavior.append(dict(**identity,**{k:v for k,v in stats.items() if not k.startswith("oracle_")}))
                        if kind == "regression":
                            for index,case in sorted(enumerate(cases),key=lambda item:item[1]["squared_error_change"],reverse=True)[:5]:
                                if case["squared_error_change"] > 0:
                                    harms.append(dict(**identity,query_index=index,retrieval=float(a[index]),final=float(b[index]),
                                        target=float(y[index]),harm_share=case["squared_error_change"]/stats["total_harm_sse"],**case))
            print(f"[read] {task} s{seed}",flush=True)
    if missing:
        raise ValueError(f"{len(missing)} missing units: {missing[:5]}")
    expected = len(tasks)*len(seeds)*27
    if len(rows) != expected or len({(r["task"],r["seed"],r["method"]) for r in rows}) != expected:
        raise AssertionError("incomplete or duplicated matrix")
    selected_rates,selected_both,zero_selected = [],[],[]
    for task in tasks:
        for seed in seeds:
            for adapter in s.DESIGNS:
                candidates = [r for r in rows if r["task"]==task and r["seed"]==seed and r["adapter"]==adapter]
                selected_both.append(policy_row(choose(candidates),"both"))
                zero_selected.append(policy_row(choose([r for r in candidates if r["dropout"]==0]),"zero_scheme"))
                for scheme in s.SCHEMES:
                    selected_rates.append(policy_row(choose([r for r in candidates if r["scheme"]==scheme]),"rate"))
    task_rows,families = aggregate(rows,tasks,seeds)
    selected_rate_tasks,selected_rate_families = aggregate(selected_rates,tasks,seeds)
    selected_both_tasks,selected_both_families = aggregate(selected_both,tasks,seeds)
    zero_selected_tasks,zero_selected_families = aggregate(zero_selected,tasks,seeds)
    all_tasks = task_rows+selected_rate_tasks+selected_both_tasks+zero_selected_tasks
    all_runs = rows+selected_rates+selected_both+zero_selected
    all_families = families+selected_rate_families+selected_both_families+zero_selected_families
    comparisons = []
    for kind in ("classification","regression"):
        for adapter in s.DESIGNS:
            for scheme in s.SCHEMES:
                zero_method = method(adapter,scheme,0.)
                for candidate in [method(adapter,scheme,p) for p in RATES[1:]]+[f"{scheme}__{adapter}__val_rate"]:
                    pair = pair_summary(all_tasks,all_runs,candidate,zero_method,kind)
                    if pair: comparisons.append(pair)
            pair = pair_summary(all_tasks,all_runs,f"val_both__{adapter}",f"val_scheme__{adapter}__p00",kind)
            if pair: comparisons.append(pair)
    # Adaptation behavior of selected policies is traced back to the chosen raw unit.
    by_unit = {b["unit"]:b for b in behavior}
    all_behavior = list(behavior)
    for chosen in selected_rates+selected_both+zero_selected:
        all_behavior.append({**by_unit[chosen["unit"]], **{k:chosen[k] for k in
            ("method","scheme","rate_setting","selected_dropout","selected_scheme")}})
    behavior_totals,budgets,selection_counts = [],[],[]
    for family in all_families:
        kind,name = family["task_type"],family["method"]
        identity = {k:family[k] for k in ("task_type","method","adapter","scheme","rate_setting")}
        part = [b for b in all_behavior if b["task_type"]==kind and b["method"]==name]
        total = dict(identity)
        for field in part[0]:
            if field.startswith("count_") or field in {"n","improved","worsened","pre_wrong","post_wrong","margin_improves","margin_worsens"} or field.endswith("_sse"):
                total[field] = sum(b[field] for b in part)
        behavior_totals.append(total)
        part = [r for r in all_runs if r["task_type"]==kind and r["method"]==name]
        budget = dict(identity)
        for field in ("epochs_run","selected_epoch","retrieval_updates_executed","retrieval_updates_selected","adapter_updates_executed","adapter_updates_selected"):
            budget["mean_"+field] = float(np.mean([r[field] for r in part]))
            budget["total_"+field] = sum(r[field] for r in part)
        budgets.append(budget)
        if any("selected_dropout" in r for r in part):
            selection_counts.append(dict(**identity,n_runs=len(part),
                **{rate_tag(p):sum(r["dropout"]==p for r in part) for p in RATES},
                **{scheme:sum(r["selected_scheme"]==scheme for r in part) for scheme in s.SCHEMES}))
    report = dict(study=study,protocol=PROTOCOL,tasks_included=list(tasks),seeds=list(seeds),checks=checks,
        new_units=sum(not r["reused"] for r in rows),reused_units=sum(r["reused"] for r in rows),
        rows=rows,tasks=task_rows,families=families,selected_rates=selected_rates,selected_both=selected_both,
        zero_selected=zero_selected,selected_rate_tasks=selected_rate_tasks,selected_rate_families=selected_rate_families,
        selected_both_tasks=selected_both_tasks,selected_both_families=selected_both_families,
        zero_selected_tasks=zero_selected_tasks,zero_selected_families=zero_selected_families,
        comparisons=comparisons,selection_counts=selection_counts,behavior=behavior,
        behavior_totals=behavior_totals,harm_examples=harms,budgets=budgets)
    out = REPO/"results"/"adaptation_improvement"/study
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name in ("rows","tasks","families","selected_rates","selected_both","zero_selected",
                 "selected_rate_tasks","selected_rate_families","selected_both_tasks","selected_both_families",
                 "zero_selected_tasks","zero_selected_families","comparisons","selection_counts","behavior",
                 "behavior_totals","harm_examples","budgets"):
        s.write_csv(out/(name+".csv"),report[name])
    plot_summary(report,out)
    print(json.dumps(dict(rows=len(rows),new=report["new_units"],reused=report["reused_units"],checks=checks,
        summary=str(out/"comparison.json"))),flush=True)
    return report


def plot_summary(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [k for k in ("classification","regression") if any(r["task_type"]==k for r in report["families"])]
    labels = {"original":"Original", "query_aware":"Query + prediction", "query_no_prediction":"Query without prediction"}
    fig,axes = plt.subplots(3,len(kinds),figsize=(5.8*len(kinds),9.5),layout="constrained",squeeze=False)
    for col,kind in enumerate(kinds):
        factor = 100 if kind=="classification" else 1
        all_values = [r["mean_error"]*factor for r in report["families"] if r["task_type"]==kind]
        low,high = min(all_values),max(all_values)
        for row,adapter in enumerate(s.DESIGNS):
            values = np.array([[next(r["mean_error"] for r in report["families"]
                if r["task_type"]==kind and r["method"]==method(adapter,scheme,p))*factor
                for scheme in s.SCHEMES] for p in RATES])
            ax = axes[row,col]
            ax.imshow(values,cmap="YlOrRd",vmin=low,vmax=high,aspect="auto")
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,f"{values[i,j]:.3f}" if kind=="classification" else f"{values[i,j]:.5f}",
                        ha="center",va="center",color="white" if values[i,j]>(low+high)/2 else "black")
            ax.set_xticks(range(3),[s.capitalize() for s in s.SCHEMES])
            ax.set_yticks(range(3),["0% dropout","10% dropout","20% dropout"])
            ax.set_title(labels[adapter]+(" | Class error (%)" if kind=="classification" else " | Standardized RMSE"),fontsize=10)
    fig.suptitle("Hidden dropout: equal task means over three seeds; lower is better")
    fig.savefig(out/"dropout_means.png",dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=q.prior.TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=q.prior.SEEDS)
    parser.add_argument("--adapters", nargs="+", choices=tuple(s.DESIGNS), default=list(s.DESIGNS))
    parser.add_argument("--schemes", nargs="+", choices=s.SCHEMES, default=list(s.SCHEMES))
    parser.add_argument("--rates", nargs="+", type=float, choices=RATES, default=[.1,.2])
    args = parser.parse_args(argv)
    records.check_study(args.study)
    torch.set_num_threads(1)
    if args.mode == "summarize":
        return summarize(args.study, args.tasks, args.seeds)
    records.check_unit_paths(args.study, [(EXPERIMENT, task.split("__",1)[1], f"{scheme[:3]}_{s.TAGS[a]}_{rate_tag(p)}", f"s{seed}")
        for task in args.tasks for seed in args.seeds for a in args.adapters for scheme in args.schemes for p in args.rates])
    for task in args.tasks:
        for seed in args.seeds:
            refs, sources = s.references(task, seed)
            for adapter in args.adapters:
                for rate in args.rates:
                    for scheme in args.schemes:
                        run_one(args.study, task, seed, adapter, scheme, rate, refs, sources)


if __name__ == "__main__":
    main()
