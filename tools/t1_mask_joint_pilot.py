"""Factorial mask x retrieval-freezing pilot through the existing T1 synchronization trainer."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["NNKNN_DEVICE"] = "cpu"

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.common.artifacts import environment_info, git_state, source_fingerprint, write_json
from model.common.core import CoreConfig, build_model, clone_optimizer, train_retrieval
from model.t1.calibration import calibrate_free_radius
from model.t1.sync import SyncConfig, _forward_terms, frozen_parameters, train_synchronized
from tools.t1_pre_topk_pilot import data_for, state_digest


@torch.no_grad()
def measure(model, adapter, X, y, cfg, radius):
    was_m, was_a = model.training, None if adapter is None else adapter.training
    model.eval()
    if adapter is not None:
        adapter.eval()
    r = model.retrieve(X, exclude_identical=False)
    w, labels = r["weights"], model.labels[r["case_indices"]].float()
    pre = w @ labels
    result = {"target": y.tolist(), "effective_neighbors": float((1 / w.square().sum(1)).mean())}
    if model.task_type == "classification":
        pre = pre / pre.sum(1, keepdim=True).clamp_min(1e-12)
        result.update(pre_error=float((pre.argmax(1) != y).float().mean()),
                      pre_loss=float(F.nll_loss(pre.clamp_min(1e-8).log(), y)),
                      pre_prediction=pre.argmax(1).tolist(), pre_probability=pre.tolist())
        if adapter is not None:
            _, scores = adapter(r["query_features"] - w @ r["case_features"], pre, None)
            logits = cfg.score_scale * scores
            result.update(post_error=float((scores.argmax(1) != y).float().mean()),
                          post_loss=float(F.cross_entropy(logits, y)), post_prediction=scores.argmax(1).tolist(),
                          post_logits=logits.tolist())
    else:
        pre = pre.view(-1)
        loss = F.mse_loss(pre, y)
        result.update(pre_error=float(loss.sqrt()), pre_loss=float(loss), pre_prediction=pre.tolist())
        if adapter is not None:
            post = adapter.forward_aggregate(r["query_features"], r["case_features"], labels, w).view(-1)
            loss = F.mse_loss(post, y)
            result.update(post_error=float(loss.sqrt()), post_loss=float(loss), post_prediction=post.tolist())
    if adapter is not None:
        terms = _forward_terms(model, adapter, X, y, radius, output_mode=cfg.output_mode,
                               near_scale=1., score_scale=cfg.score_scale)
        assert np.isclose(result["pre_loss"], float(terms["L_pre"]), rtol=1e-6, atol=1e-7)
        assert np.isclose(result["post_loss"], float(terms["L_post"]), rtol=1e-6, atol=1e-7)
    model.train(was_m)
    if adapter is not None:
        adapter.train(was_a)
    return result


def continuation(base, trained, data, cc, sc, radius, frozen):
    model = copy.deepcopy(base)
    start_hash = state_digest(model)
    optimizer = clone_optimizer(trained.optimizer, model, cc)
    captured = {}

    def observe(epoch, m, a):
        # Read-only snapshots: the trainer still selects exclusively by validation loss.
        captured["last"] = (copy.deepcopy(m.state_dict()), copy.deepcopy(a.state_dict()))
        if epoch == sc.adapter_warmup_epochs:
            captured["warmup"] = {"retriever": state_digest(m), "adapter": state_digest(a)}
            assert captured["warmup"]["retriever"] == start_hash
        return {}

    # Same trainer, objectives, warm-up, minibatch seed and Adam states in both arms.
    # Freezing only retrieval parameters makes the frozen arm a matched sequential control.
    with frozen_parameters(model) if frozen else nullcontext():
        adapter, info = train_synchronized(model, data, cc, sc, radius, near_scale=1.,
                                           core_optimizer=optimizer, epoch_observer=observe)
    selected = measure(model, adapter, data.X_test, data.y_test, sc, radius)
    selected_hash = state_digest(model)
    last_model, last_adapter = copy.deepcopy(model), copy.deepcopy(adapter)
    last_model.load_state_dict(captured["last"][0])
    last_adapter.load_state_dict(captured["last"][1])
    last = measure(last_model, last_adapter, data.X_test, data.y_test, sc, radius)
    last_hash = state_digest(last_model)
    if frozen:
        assert start_hash == selected_hash == last_hash, "Frozen retriever changed"
    return {"frozen": frozen, "pretrained_state_sha256": start_hash,
            "warmup_end_sha256": captured["warmup"], "selected_retriever_sha256": selected_hash,
            "last_retriever_sha256": last_hash, "selected": selected, "last": last,
            "best_epoch": info["best_epoch"], "epochs_run": len(info["history"]),
            "selected_postwarmup_epochs": max(0, info["best_epoch"] - sc.adapter_warmup_epochs),
            "history": info["history"], "sync": dataclasses.asdict(sc)}


def aggregate(rows):
    out = []
    for name in dict.fromkeys(r["dataset"] for r in rows):
        task = {"dataset": name, "task_type": next(r["task_type"] for r in rows if r["dataset"] == name)}
        for masked, label in [(True, "masked"), (False, "all_cases")]:
            for frozen, schedule in [(True, "frozen"), (False, "joint")]:
                rs = [r for r in rows if (r["dataset"], r["masked"], r["frozen"]) == (name, masked, frozen)]
                values = {"baseline_error": [r["baseline"]["pre_error"] for r in rs],
                          "baseline_loss": [r["baseline"]["pre_loss"] for r in rs],
                          "pre_error": [r["selected"]["pre_error"] for r in rs],
                          "post_error": [r["selected"]["post_error"] for r in rs],
                          "pre_loss": [r["selected"]["pre_loss"] for r in rs],
                          "post_loss": [r["selected"]["post_loss"] for r in rs],
                          "loss_drift": [r["selected"]["pre_loss"] - r["baseline"]["pre_loss"] for r in rs],
                          "last_loss_drift": [r["last"]["pre_loss"] - r["baseline"]["pre_loss"] for r in rs]}
                task[label + "_" + schedule] = {"n": len(rs), **{k: float(np.mean(v)) for k, v in values.items()},
                    "loss_drift_sd": float(np.std(values["loss_drift"], ddof=1)) if len(rs) > 1 else None,
                    "selected_after_warmup": sum(r["selected_postwarmup_epochs"] > 0 for r in rs)}
        task["selected_loss_interaction"] = task["all_cases_joint"]["loss_drift"] - task["masked_joint"]["loss_drift"]
        task["last_loss_interaction"] = task["all_cases_joint"]["last_loss_drift"] - task["masked_joint"]["last_loss_drift"]
        out.append(task)
    return out


def report(study, rows, out, config):
    summary = aggregate(rows)
    write_json(out / "summary.json", summary)
    lines = ["# Mask x joint-training pilot, October 30, 2026", "",
             f"Exploratory 2 x 2 comparison on {len(study['datasets'])} small legacy datasets and seeds {study['seeds']}: "
             f"{len(rows)//2} retrieval pretrains and {len(rows)} frozen/joint continuations.", "",
             "A = masked + frozen; B = masked + joint; C = all-case + frozen; D = all-case + joint. "
             "Within each mask condition, frozen and joint arms share an identical pretrained checkpoint and Adam state, "
             "adapter initialization, five-epoch adapter warm-up and minibatch order. The warm-up endpoint hashes are checked. "
             "Both use the same summed objective after warm-up; only retrieval trainability differs. This matched frozen "
             "control is not an exact replication of the historical sequential arm's adapter-only objective.", "",
             "Only training/validation streams influence fitting and checkpoint selection. Identity leave-one-out is "
             "retained in training. Bias initialization uses bias_init_k=5 in both mask conditions; masking uses top_k=5. "
             "Raw features, one weight set, tau=1. No retention, guard, encoder, or tuning. Regression preprocessing uses "
             "model.regression_workflow; Student Performance's category vocabulary is fitted on train only.", "",
             f"Pretraining: at most {study['core']['epochs']} epochs, patience {study['core']['patience']}; "
             f"continuation: {study['sync']['epochs']} epochs including {study['sync']['adapter_warmup_epochs']} warm-up, "
             f"patience {study['sync']['patience']}. Best checkpoint uses validation post-adaptation loss, including warm-up. "
             "Last executed checkpoint is also recorded to expose deterioration that model selection may discard.", "",
             "## Standalone retrieval error at the selected checkpoint", "",
             "Lower is better. Classification uses error rate (fraction); regression uses standardized-target RMSE. "
             "Frozen-arm retrieval equals its own pretrained reference exactly.", "",
             "| Dataset | Masked pretrained | B: masked joint retrieval | All-case pretrained | D: all-case joint retrieval |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for s in summary:
        m, u = s["masked_joint"], s["all_cases_joint"]
        lines.append(f"| {s['dataset']} | {m['baseline_error']:.4f} | {m['pre_error']:.4f} | {u['baseline_error']:.4f} | {u['pre_error']:.4f} |")
    lines += ["", "## Raw retrieval-loss change relative to each arm's own pretrained reference", "",
              "Classification uses NLL with the implementation's 1e-8 probability floor; regression uses MSE. "
              "Positive drift means worse retrieval. Interaction = all-case joint drift minus masked joint drift; "
              "negative values favor all-case for preserving retrieval loss. Do not pool these raw units across tasks. "
              "A negative interaction alone is not proof that damage was prevented: the masked arm may itself improve.", "",
              "| Dataset | Selected masked drift | Selected all-case drift | Selected interaction | Last masked drift | Last all-case drift |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s in summary:
        m, u = s["masked_joint"], s["all_cases_joint"]
        lines.append(f"| {s['dataset']} | {m['loss_drift']:+.5f} | {u['loss_drift']:+.5f} | {s['selected_loss_interaction']:+.5f} | "
                     f"{m['last_loss_drift']:+.5f} | {u['last_loss_drift']:+.5f} |")
    lines += ["", "![Paired retrieval-loss drift](retrieval_loss_drift.png)", "",
              "Dots show each seed, diamonds show means; thin lines pair the same seed. Panels use separate scales. "
              "The dashed zero line is no change from the pretrained reference. Values are not confidence intervals.", "",
              "## Final adapted error at the selected checkpoint", "",
              "| Dataset | A: masked frozen | B: masked joint | C: all-case frozen | D: all-case joint | B selected after warm-up | D selected after warm-up |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for s in summary:
        arms = [s[k] for k in ["masked_frozen", "masked_joint", "all_cases_frozen", "all_cases_joint"]]
        lines.append(f"| {s['dataset']} | " + " | ".join(f"{a['post_error']:.4f}" for a in arms)
                     + f" | {arms[1]['selected_after_warmup']}/{arms[1]['n']} | {arms[3]['selected_after_warmup']}/{arms[3]['n']} |")
    lines += ["", "This small untuned pilot tests masking throughout the pipeline, including pretraining. It does not "
              "isolate toggling masking only after a common cross-mask pretrain, prove the proposed neighborhood-jump "
              "mechanism, or establish a general result beyond these datasets/seeds. Selected and last checkpoints answer "
              "different questions. CPU, one thread, deterministic algorithms; actual runtime versions are in the manifest.", "",
              f"Configuration: [YAML](../../../../{Path(config).as_posix()}). Evidence: [rows](runs.json), "
              "[summary JSON](summary.json), [manifest](manifest.json).", "",
              "```powershell", f".venv/Scripts/python.exe tools/t1_mask_joint_pilot.py --config {Path(config).as_posix()} --out results/t1/p4_component_sync/mask_joint_rerun", "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots((len(summary)+2)//3, 3, figsize=(13, 3.5*((len(summary)+2)//3)), squeeze=False, constrained_layout=True)
    for ax, s in zip(axes.flat, summary):
        for seed in study["seeds"]:
            rs = [next(r for r in rows if r["dataset"] == s["dataset"] and r["seed"] == seed and r["masked"] == m and not r["frozen"])
                  for m in [True, False]]
            values = [r["selected"]["pre_loss"]-r["baseline"]["pre_loss"] for r in rs]
            ax.plot([0,1], values, color="#b8b8b8", linewidth=1)
            ax.scatter([0,1], values, c=["#245e8b", "#bb7619"], zorder=3)
        ax.scatter([0,1], [s["masked_joint"]["loss_drift"], s["all_cases_joint"]["loss_drift"]],
                   marker="D", c=["#245e8b", "#bb7619"], edgecolor="black", zorder=4)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks([0,1], ["Masked joint", "All-case joint"])
        ax.set_xlim(-.25,1.25)
        ax.margins(y=.15)
        ax.set_ylabel("Change in retrieval NLL" if s["task_type"] == "classification" else "Change in retrieval MSE")
        ax.set_title(s["dataset"].replace("_", " ").title())
        ax.spines[["top","right"]].set_visible(False)
    for ax in list(axes.flat)[len(summary):]:
        ax.set_visible(False)
    fig.suptitle("Joint fine-tuning: retrieval-loss drift at the validation-selected checkpoint\nPositive is deterioration; diamonds are means")
    fig.savefig(out / "retrieval_loss_drift.png", dpi=160)
    plt.close(fig)


def run(study, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Output already exists; choose a fresh directory")
    (out / "runner_at_execution.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(out / "manifest.json", {"study": study, "git": git_state(REPO), "source_sha256": source_fingerprint(REPO),
                                       "environment": environment_info(), "actual_python": sys.executable,
                                       "note": "Local environment; no historical bitwise-reproduction claim."})
    rows = []
    started = time.perf_counter()
    for ds in study["datasets"]:
        for seed in study["seeds"]:
            data = data_for(ds, seed, study["splits"])
            initial_hash = None
            for masked in [True, False]:
                cc = CoreConfig(task_type=ds["task_type"], seed=seed, pre_topk_mask=masked, **study["core"])
                base = build_model(data.X_train, data.y_train, cc, data.num_classes)
                initial_hash = initial_hash or state_digest(base)
                assert state_digest(base) == initial_hash
                trained = train_retrieval(base, data.X_train, data.y_train, data.X_val, data.y_val, cc)
                radius, _ = calibrate_free_radius(base, s_task=.5, snapshot_step=trained.best_epoch)
                sync = {**study["sync"], "hidden_dims": tuple(study["sync"]["hidden_dims"])}
                sc = SyncConfig(seed=seed+10000, output_mode="aggregate_regression" if ds["task_type"] == "regression" else "nominal_residual_scores",
                                score_scale=study.get("classification_score_scale",10.) if ds["task_type"] == "classification" else 1., **sync)
                baseline = measure(base, None, data.X_test, data.y_test, sc, radius)
                paired_warmup = None
                for frozen in [True, False]:
                    r = continuation(base, trained, data, cc, sc, radius, frozen)
                    paired_warmup = paired_warmup or r["warmup_end_sha256"]
                    assert r["warmup_end_sha256"] == paired_warmup
                    if frozen:
                        assert r["selected"]["pre_loss"] == baseline["pre_loss"]
                        assert r["selected"]["pre_prediction"] == baseline["pre_prediction"]
                    r.update(dataset=ds["name"], task_type=ds["task_type"], seed=seed, masked=masked,
                             initial_state_sha256=initial_hash, core=dataclasses.asdict(cc), baseline=baseline,
                             bias_initialization=base.t1_bias_init, data=data.describe(), free_radius=radius.to_dict(),
                             pretrain_best_epoch=trained.best_epoch, pretrain_epochs_run=trained.epochs_run,
                             pretrain_history=trained.history)
                    rows.append(r)
                    write_json(out / "runs" / ds["name"] / f"s{seed}" / f"mask{int(masked)}_frozen{int(frozen)}.json", r)
                    write_json(out / "runs.json", rows)
                    print(f"{ds['name']} seed={seed} mask={masked} frozen={frozen} "
                          f"retrieval={r['selected']['pre_error']:.5f} adapted={r['selected']['post_error']:.5f} "
                          f"loss_drift={r['selected']['pre_loss']-baseline['pre_loss']:+.5f} best={r['best_epoch']}", flush=True)
    report(study, rows, out, config)
    print(f"Completed {len(rows)} continuations in {time.perf_counter()-started:.1f}s: {out/'report.md'}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/t1/mask_joint_pilot_20261030.yaml")
    parser.add_argument("--out", default="results/t1/p4_component_sync/mask_joint_pilot_20261030")
    args = parser.parse_args()
    study = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(study, args.out, args.config)


if __name__ == "__main__":
    main()
