"""Learn one disagreement cutoff and compare with the completed ag1 methods."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import torch
from model.common import tracking
from model.t1 import records
from model.t1.adaptation_gate import outcome
from model.t1.disagreement_threshold import DisagreementThresholdAdapter, MODES, train_threshold
from tools import t1_adaptation_gate as prior

EXPERIMENT = "threshold_v1"
PROTOCOL = dict(version=1, epochs=60, patience=12, batch_size=64, temperature=.25,
                threshold_lr=.05, threshold_steps=5, disagreement_std_floor=1e-6,
                initialization="training_median", normalization="training_mean_std",
                routing_cost="classification_01_regression_mse", routing_gradient="detached_branch_costs",
                frozen_selection="final_training_iterate", joint_selection=prior.PROTOCOL["selection"],
                retrieval="frozen_old_selected_core", train_queries="stable_id_leave_one_out")
METHODS = (*prior.STATIC, *prior.MODES, *MODES)


def key_for(study, task, seed, mode):
    return records.key(study, EXPERIMENT, task.split("__", 1)[1], mode, f"s{seed}")


def reference_units(task, seed, reference_study, source_study):
    _, _, source, fp = prior.source_for(task, seed, source_study)
    units = {}
    for arm in ("inputs", "controls", *prior.MODES):
        config = prior.config_for(task, seed, source_study, source, fp, arm)
        key = prior.key_for(reference_study, task, seed, arm)
        unit = tracking.load_unit(key, config=config, arrays=False)
        if unit is None:
            raise ValueError(f"completed reference unit required: {key.run_id}")
        units[arm] = unit
    return units


def config_for(task, seed, mode, inputs):
    return dict(experiment=EXPERIMENT, task=task, seed=seed, mode=mode,
                source_unit=inputs.info.run_id,
                source_config_identity=tracking.config_identity(inputs.config),
                data_fingerprint=inputs.config["data_fingerprint"], protocol=PROTOCOL)


def run_one(study, task, seed, mode, inputs, cache):
    config = config_for(task, seed, mode, inputs)
    key = key_for(study, task, seed, mode)
    if tracking.load_unit(key, config=config, arrays=False) is not None:
        print(f"[recorded] {task} s{seed} {mode}", flush=True)
        return
    with tracking.open_run(key, config=config) as run:
        model = DisagreementThresholdAdapter(records.unpack(cache["network"]), cache["task_type"],
                                              mode, temperature=PROTOCOL["temperature"])
        settings = {**cache["settings"], **{k:PROTOCOL[k] for k in
                    ("epochs", "patience", "batch_size", "threshold_lr", "threshold_steps")}}
        values = cache["values"]
        info = train_threshold(model, values["train"], values["val"], settings, seed, run=run)
        row = dict(task=task, task_type=cache["task_type"], seed=seed, method=mode, **info)
        for split in ("train", "val", "test"):
            result, p = outcome(model, values[split], settings)
            row.update({split+"_"+k:v for k,v in result.items()})
            prior.record_prediction(run, split+"_predictions", values[split], p.final, p.apply, p.full, p.logit)
        run.log_records("rows", [records.ordered(row)])
        run.log_final({k:v for k,v in row.items() if isinstance(v, (int, float))})
        run.log_record("source_reference", dict(unit=inputs.info.run_id, adapter=cache["source_unit"]))
        records.save_rng(run)
    print(f"[done] {task} s{seed} {mode}: error={row['test_error']:.6f}, "
          f"apply={row['test_apply_rate']:.3f}, threshold={row['threshold']:.6f}, "
          f"epoch={row['selected_epoch']}", flush=True)


def run_task(study, task, seeds, reference_study, source_study):
    records.check_unit_paths(study, [(EXPERIMENT, task.split("__", 1)[1], m, f"s{s}")
                                   for s in seeds for m in MODES])
    for seed in seeds:
        refs = reference_units(task, seed, reference_study, source_study)
        complete = [tracking.load_unit(key_for(study, task, seed, m),
                    config=config_for(task, seed, m, refs["inputs"]), arrays=False) is not None for m in MODES]
        if all(complete):
            print(f"[recorded] {task} s{seed} both thresholds", flush=True)
            continue
        cache = refs["inputs"].load_state("inputs")
        for mode in MODES:
            run_one(study, task, seed, mode, refs["inputs"], cache)


def compare(part, reference):
    deltas = [a["test_error"]-b["test_error"] for a,b in zip(part, reference)]
    positive = [(a,b) for a,b in zip(part, reference) if min(a["test_error"], b["test_error"])>0]
    return dict(wins=sum(d < -1e-7 for d in deltas), ties=sum(abs(d)<=1e-7 for d in deltas),
                losses=sum(d > 1e-7 for d in deltas),
                gmr=float(np.exp(np.mean([np.log(a["test_error"]/b["test_error"]) for a,b in positive]))) if positive else None,
                zero_exclusions=[a["task"] for a,b in zip(part, reference) if min(a["test_error"], b["test_error"])<=0])


def aggregate(rows, tasks, seeds):
    means = []
    counts = ("_n", "_helped", "_harmed", "_wrong_direction", "_harmful_overshoot")
    for task in tasks:
        for method in METHODS:
            sub = [r for r in rows if r["task"] == task and r["method"] == method]
            if len(sub) != len(seeds):
                continue
            item = dict(task=task, task_type=sub[0]["task_type"], method=method, n_seeds=len(sub))
            for k in sorted(set.intersection(*(set(r) for r in sub))):
                if k.startswith(("train_", "val_", "test_")) and all(isinstance(r[k], (int, float)) for r in sub):
                    item[k] = sum(r[k] for r in sub) if k.endswith(counts) else float(np.mean([r[k] for r in sub]))
            means.append(item)
    families = []
    for kind in ("classification", "regression"):
        for method in METHODS:
            part = [r for r in means if r["task_type"] == kind and r["method"] == method]
            if not part:
                continue
            item = dict(task_type=kind, method=method, n_tasks=len(part),
                        mean_error=float(np.mean([r["test_error"] for r in part])),
                        mean_apply_rate=float(np.mean([r["test_apply_rate"] for r in part])),
                        helped=sum(r["test_helped"] for r in part), harmed=sum(r["test_harmed"] for r in part),
                        n=sum(r["test_n"] for r in part),
                        applied=sum(r["test_n"]*r["test_apply_rate"] for r in part))
            # Different seeds can have different test sizes: pool original rows for application count.
            raw = [r for r in rows if r["task_type"] == kind and r["method"] == method]
            item["applied"] = int(round(sum(r["test_n"]*r["test_apply_rate"] for r in raw)))
            item["always_apply_runs"] = sum(r["test_apply_rate"] == 1. for r in raw)
            item["never_apply_runs"] = sum(r["test_apply_rate"] == 0. for r in raw)
            if all("selected_epoch" in r for r in raw):
                item["epoch_zero_runs"] = sum(r["selected_epoch"] == 0 for r in raw)
            if all("train_benefit_fraction" in r for r in raw):
                item["train_no_routing_signal_runs"] = sum(r["train_benefit_fraction"] == 0.
                    and r["train_harm_fraction"] == 0. for r in raw)
            for field in ("test_full_error", "test_wrong_direction", "test_harmful_overshoot"):
                if all(field in r for r in part):
                    item[field] = float(np.mean([r[field] for r in part])) if field.endswith("error") else sum(r[field] for r in part)
            for reference in ("original", "disagreement_gate", "full_control"):
                baseline = [next(r for r in means if r["task"]==p["task"] and r["method"]==reference) for p in part]
                item["vs_"+reference] = compare(part, baseline)
            families.append(item)
    return means, families


def verify_prediction(unit, row, split, *, frozen_original=None, inputs=None, checkpoint=None):
    data = unit.load_state(split+"_predictions")
    target, final, p0, full, apply = (data[k] for k in ("target", "final", "retrieval", "full", "apply"))
    expected = torch.where(apply[:, None], full, p0)
    if not torch.equal(final, expected):
        raise AssertionError("threshold output is not exactly full-or-skip")
    cls = row["task_type"] == "classification"
    error = float((final.argmax(1)!=target).float().mean()) if cls else float((final-target.reshape_as(final)).square().mean().sqrt())
    if not np.isclose(error, row[split+"_error"], atol=1e-7):
        raise AssertionError("saved prediction/metric mismatch")
    if not np.isclose(float(apply.float().mean()), row[split+"_apply_rate"], atol=1e-7):
        raise AssertionError("saved application rate mismatch")
    if inputs is not None:
        if not torch.equal(target, inputs["target"]) or not torch.equal(p0, inputs["p0"]):
            raise AssertionError("query targets or retrieval differ from the reference")
        z = (inputs["statistics"][:, 3]-checkpoint["disagreement_mean"])/checkpoint["disagreement_std"]
        if not torch.equal(apply, z > checkpoint["threshold"]):
            raise AssertionError("saved decisions differ from the learned scalar cutoff")
    if cls:
        before, after = p0.argmax(1)!=target, final.argmax(1)!=target
        helped, harmed = before & ~after, ~before & after
    else:
        before = (p0-target.reshape_as(p0)).square().mean(1)
        after = (final-target.reshape_as(final)).square().mean(1)
        helped, harmed = after < before-1e-8, after > before+1e-8
    if int(helped.sum()) != row[split+"_helped"] or int(harmed.sum()) != row[split+"_harmed"]:
        raise AssertionError("saved helpful/harmful correction counts mismatch")
    if frozen_original is not None and not torch.equal(full, frozen_original):
        raise AssertionError("frozen threshold changed the original full proposal")
    return len(target)


def summarize(study, tasks, seeds, reference_study, source_study):
    rows, missing, checks = [], [], 0
    for task in tasks:
        for seed in seeds:
            refs = reference_units(task, seed, reference_study, source_study)
            inputs = refs["inputs"].load_state("inputs")["values"]
            for arm in ("controls", *prior.MODES):
                rows.extend(records.rows(refs[arm], "rows"))
            for mode in MODES:
                key = key_for(study, task, seed, mode)
                saved = tracking.load_unit(key, config=config_for(task, seed, mode, refs["inputs"]), arrays=False)
                if saved is None:
                    missing.append(key.run_id)
                    continue
                row, = records.rows(saved, "rows")
                checkpoint = saved.load_state("checkpoint")["model"]
                for split in ("train", "val", "test"):
                    original = refs["controls"].load_state("original_"+split)["full"] if mode=="threshold_frozen" and split!="train" else None
                    verify_prediction(saved, row, split, frozen_original=original,
                                      inputs=inputs[split], checkpoint=checkpoint)
                    checks += 1
                rows.append(row)
    if len({(r["task"], r["seed"], r["method"]) for r in rows}) != len(rows):
        raise AssertionError("duplicate comparison row")
    means, families = aggregate(rows, tasks, seeds)
    report = dict(study=study, reference_study=reference_study, source_study=source_study,
                  protocol=PROTOCOL, missing=missing, prediction_checks=checks,
                  rows=rows, tasks=means, families=families)
    out = REPO/"results"/"adaptation_improvement"/study
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for name, items in (("runs", rows), ("tasks", means), ("families", families)):
        if items:
            with (out/(name+".csv")).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in items for k in r)))
                writer.writeheader()
                writer.writerows(items)
    if not missing:
        plot(report, out)
    print(json.dumps(dict(rows=len(rows), missing=len(missing), checks=checks,
                         summary=str(out/"comparison.json"))), flush=True)
    return report


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [kind for kind in ("classification", "regression")
             if any(r["task_type"] == kind for r in report["families"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7*len(kinds), 5), layout="constrained", squeeze=False, sharey=True)
    labels = {"original":"Original full adapter", "disagreement_gate":"Validation threshold",
              "threshold_frozen":"Learned threshold / frozen adapter", "threshold_joint":"Learned threshold + adapter",
              "full_control":"Extra training, no gate", "probability_gate":"Probability gate", "utility_gate":"Utility gate"}
    for ax, kind in zip(axes[0], kinds):
        part = {r["method"]:r for r in report["families"] if r["task_type"]==kind}
        values = [part[m]["mean_error"]*(100 if kind=="classification" else 1) for m in labels]
        colors = ["#24797d" if m.startswith("threshold_") else "#a5b4c1" for m in labels]
        ax.barh(list(labels.values()), values, color=colors)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.set_xlabel("Mean classification error (%)" if kind=="classification" else "Mean standardized RMSE")
        n_tasks = part["original"]["n_tasks"]
        n_seeds = next(r["n_seeds"] for r in report["tasks"] if r["task_type"] == kind)
        ax.set_title(f"{kind.capitalize()} / {n_tasks} tasks, {n_seeds} seeds")
        ax.set_xlim(0, max(values)*1.25)
        for i, value in enumerate(values):
            ax.text(value, i, "  "+(f"{value:.3f}" if kind=="classification" else f"{value:.6f}"), va="center", fontsize=9)
    fig.suptitle("Learning the disagreement threshold: equal-task mean test error")
    fig.savefig(out/"threshold_comparison.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--reference-study", default="ag1")
    parser.add_argument("--source-study", default="am1")
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--tasks", nargs="+", default=prior.TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(prior.SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.tasks))!=len(args.tasks) or not set(args.tasks)<=set(prior.TASKS):
        parser.error("choose unique local tasks")
    if len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(prior.SEEDS):
        parser.error("choose unique seeds 40-42")
    if args.study in (args.reference_study, args.source_study):
        parser.error("use a new study name for the threshold experiment")
    records.check_study(args.study)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == "run":
        for task in args.tasks:
            run_task(args.study, task, args.seeds, args.reference_study, args.source_study)
    else:
        return summarize(args.study, args.tasks, args.seeds, args.reference_study, args.source_study)


if __name__ == "__main__":
    main()
