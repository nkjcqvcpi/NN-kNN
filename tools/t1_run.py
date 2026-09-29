"""Run a T1 experiment from a YAML config.

    .venv\\Scripts\\python.exe tools\\t1_run.py configs\\t1\\p1_retention_pilot.yaml [--only-dataset iris] [--seeds 0 1]

Experiment types (``type:``): legacy_reference, retention, reuse, sync, mcb, revise.
Every run writes, under ``results/t1/<experiment>/<dataset>/<condition>/s<seed>/``:
run_manifest.json, metrics.json, history.json, case_statistics.jsonl,
case_maintenance.jsonl, retrieval_events.jsonl (test stream, pre-adaptation),
and appends one row to ``results/t1/<experiment>/runs__<dataset>.jsonl``.

Values required by the plan that remain unresolved PI decisions must be given
explicitly in the config; ``null`` or ``"..."`` aborts the run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from model.t1.artifacts import build_manifest, case_statistics_rows, config_id, write_json, write_jsonl  # noqa: E402
from model.t1.calibration import calibrate_free_radius  # noqa: E402
from model.t1.core import CoreConfig, build_model, clone_optimizer, evaluate, retrieval_events, train_retrieval  # noqa: E402
from model.t1.data import make_splits  # noqa: E402
from model.t1.maintenance import CaseArchive, run_maintenance  # noqa: E402
from model.t1.mcb import StabilityTracker  # noqa: E402
from model.t1.provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, audit_provenance, score_cases  # noqa: E402
from model.t1.retention import RetentionConfig  # noqa: E402
from model.t1.reuse import ReuseConfig, evaluate_reuse, train_classification_adapter  # noqa: E402
from model.t1.revise import flagging_ablation  # noqa: E402
from model.t1.sync import SyncConfig, train_synchronized  # noqa: E402


# --------------------------------------------------------------------------- config


def _check_resolved(obj: Any, path: str = "") -> None:
    if obj is None or (isinstance(obj, str) and obj.strip() == "..."):
        raise ValueError(f"Unresolved config value at '{path}'. The plan requires an explicit value (no hidden defaults).")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_resolved(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_resolved(v, f"{path}[{i}]")


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in cfg:
        base = load_config_raw(path.parent / cfg.pop("base"))
        cfg = _deep_merge(base, cfg)
    _check_resolved(cfg)
    return cfg


def load_config_raw(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in cfg:
        cfg = _deep_merge(load_config_raw(path.parent / cfg.pop("base")), cfg)
    return cfg


def core_cfg_for(cfg: dict[str, Any], task_type: str, seed: int, **over: Any) -> CoreConfig:
    c = dict(cfg["core"])
    c.update(over)
    if "mlp_dims" in c:
        c["mlp_dims"] = tuple(c["mlp_dims"])
    return CoreConfig(task_type=task_type, seed=seed, **c)


def score_cfg_for(cfg: dict[str, Any]) -> ScoreConfig:
    s = cfg["statistics"]
    return ScoreConfig(
        smoothing=float(s["smoothing"]),
        alpha=float(s["alpha"]),
        min_retrieval_count=float(s["min_retrieval_count"]),
        min_activation_mass=float(s["min_activation_mass"]),
        bias_normalization=s.get("bias_normalization", "within_cohort_percentile"),
        bias_source=s.get("bias_source", "current"),
        trust_mode=s.get("trust_mode", "geometric"),
    )


def resolve_K(k: Any, n: int) -> int:
    if isinstance(k, float) and k <= 1.0:
        return max(1, int(round(k * n)))
    return int(k)


# --------------------------------------------------------------------------- helpers


def audit(model, data, cfg, store, step):
    src = cfg["statistics"]["source"]
    if src == "train_loo":
        X, y, excl = data.X_train, data.y_train, True
    elif src == "maintenance_split":
        if data.X_maint is None:
            raise ValueError("statistics.source=maintenance_split needs splits.maint_frac > 0")
        X, y, excl = data.X_maint, data.y_maint, False
    else:
        raise ValueError(src)
    if cfg["statistics"].get("accumulation", "per_checkpoint") == "per_checkpoint":
        store.reset_counts()
    return audit_provenance(model, X, y, store, exclude_identical=excl, counterfactual=bool(cfg["statistics"]["counterfactual"]), step=step, reg_bins=data.reg_bins)


def retention_cfg_for(cond: dict[str, Any], cfg: dict[str, Any], seed: int) -> RetentionConfig:
    r = dict(cfg["retention"])
    r.update({k: v for k, v in cond.items() if k in RetentionConfig.__dataclass_fields__})
    r["seed"] = seed
    r.setdefault("notes", {})
    return RetentionConfig(**r)


def test_metrics(model, data, prefix="test"):
    e = evaluate(model, data.X_test, data.y_test)
    out = {f"{prefix}_{k}": v for k, v in e.items() if not k.startswith(("pred", "prob"))}
    if data.test_groups and model.task_type == "classification":
        pred = e["pred_pre"]
        for g, m in data.test_groups.items():
            mt = torch.as_tensor(m)
            out[f"{prefix}_accuracy_{g}"] = float((pred[mt] == data.y_test[mt]).float().mean())
    return out


def truth_retention(model, data) -> dict[str, Any]:
    if not data.case_truth:
        return {}
    kept = set(model.active_case_ids().tolist())
    out = {}
    for k, v in data.case_truth.items():
        idx = np.nonzero(v)[0]
        if idx.size:
            out[f"kept_fraction_{k}"] = float(np.mean([i in kept for i in idx]))
    clean = np.ones(len(data.y_train), dtype=bool)
    for v in data.case_truth.values():
        clean &= ~v
    out["kept_fraction_clean_base"] = float(np.mean([i in kept for i in np.nonzero(clean)[0]]))
    return out


class Runner:
    def __init__(self, cfg: dict[str, Any], out_root: Path) -> None:
        self.cfg = cfg
        self.exp = cfg["experiment"]
        self.out = out_root / self.exp
        self.out.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(cfg.get("torch_threads", 1)))

    def data_for(self, ds: dict[str, Any], seed: int):
        sp = self.cfg["splits"]
        return make_splits(ds["name"], ds["task_type"], seed, test_frac=sp["test_frac"], val_frac=sp["val_frac"], maint_frac=sp.get("maint_frac", 0.0), reg_cohort_bins=sp.get("reg_cohort_bins", 5), max_train=ds.get("max_train"))

    def trained_core(self, data, seed: int, **over):
        cc = core_cfg_for(self.cfg, data.task_type, seed, **over)
        model = build_model(data.X_train, data.y_train, cc, data.num_classes)
        tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc)
        return model, cc, tr

    def finish(self, *, ds, seed, cond_label, cond, data, model, store, archive, metrics, history, events, components, budgets, maintenance, model_desc, extra_files=None):
        run_dir = self.out / ds["name"] / cond_label / f"s{seed}"
        run_cfg = {"experiment": self.exp, "dataset": ds, "seed": seed, "condition": cond, "config": self.cfg}
        run_id = f"{self.exp}-{ds['name']}-{cond_label}-s{seed}-{config_id(run_cfg)[:8]}"
        man = build_manifest(
            run_id=run_id, repo=REPO, cfg=run_cfg, data_desc=data.describe(), components=components, budgets=budgets,
            maintenance=maintenance, evaluation={"confirmatory_or_exploratory": self.cfg["status"], "number_of_seeds": len(self.cfg["seeds"]), "uncertainty_method": "per-seed runs; summarized by tools/t1_summarize.py"}, model_desc=model_desc,
        )
        write_json(run_dir / "run_manifest.json", man)
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "history.json", history)
        if store is not None:
            write_jsonl(run_dir / "case_statistics.jsonl", case_statistics_rows(model, store, archive or CaseArchive()))
        write_jsonl(run_dir / "case_maintenance.jsonl", events or [])
        write_jsonl(run_dir / "retrieval_events.jsonl", retrieval_events(model, data.X_test, data.y_test, stream="test", run_id=run_id))
        for name, obj in (extra_files or {}).items():
            write_json(run_dir / name, obj)
        row = {"run_id": run_id, "experiment": self.exp, "dataset": ds["name"], "task_type": data.task_type, "seed": seed, "condition": cond_label, **{k: v for k, v in cond.items() if not isinstance(v, (dict, list))}, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        # one shard per dataset: parallel jobs (one per dataset) never share a file
        with (self.out / f"runs__{ds['name']}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        keys = ("test_accuracy_pre", "test_accuracy_post", "test_rmse_pre", "test_accuracy_rare", "n_cases")
        print(f"[done] {run_id} " + " ".join(f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics.get(k), (int, float))), flush=True)

    # ----------------------------------------------------------------- experiment types

    def run_legacy_reference(self, ds, seed):
        """Phase 0: the maintained NN-kNN trainer (current behavior) on the same splits."""
        from model.nnknn_model import train_model

        data = self.data_for(ds, seed)
        if data.task_type == "classification":
            from model.classification_workflow import make_classification_cfg

            lcfg = make_classification_cfg(self.cfg["legacy"].get("overrides", {}))
        else:
            from model.regression_workflow import make_regression_cfg

            lcfg = make_regression_cfg({**self.cfg["legacy"].get("overrides", {}), **self.cfg["legacy"]["regression_overrides"]})
        run_dir = self.out / ds["name"] / "legacy" / f"s{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        lcfg["checkpoint_path"] = str(run_dir / "legacy_best.pth")
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        _, _, model = train_model(data.X_train, data.y_train, data.X_val, data.y_val, None, lcfg)
        m = test_metrics(model, data)
        m["train_seconds"] = time.time() - t0
        m["n_cases"] = model.case_count()
        self.finish(ds=ds, seed=seed, cond_label="legacy", cond={"variant": "maintained_train_model"}, data=data, model=model, store=None, archive=None, metrics=m, history=[], events=[],
                    components={"retrieve": "maintained core", "reuse_or_adapter": "off", "revise": "off", "retain": "none (full memory)", "mcb": "off", "component_synchronization": "off"},
                    budgets={"case_capacity": model.case_count(), "training_epochs_max": lcfg.get("training_epochs")}, maintenance={"policy": "none"}, model_desc={"baseline_or_nnknn_variant": "legacy maintained train_model", "cfg": lcfg})

    def run_retention(self, ds, seed):
        data = self.data_for(ds, seed)
        base, cc, tr = self.trained_core(data, seed)
        base_metrics = test_metrics(base, data, "fullmem")
        proto = self.cfg["protocol"]
        for cond in self.cfg["conditions"]:
            for K_spec in self.cfg["K"]:
                model = copy.deepcopy(base)
                n = model.case_count()
                K = resolve_K(K_spec, n)
                label = f"{cond['policy']}_K{K_spec}"
                store, archive = CaseStatisticsStore(data.task_type), CaseArchive()
                rcfg = retention_cfg_for(cond, self.cfg, seed)
                scfg = score_cfg_for(self.cfg)
                opt = clone_optimizer(tr.optimizer, model, cc)  # continue the core's Adam state (see clone_optimizer)
                ainfo = audit(model, data, self.cfg, store, step=0)
                res = run_maintenance(model, store, archive, K, rcfg, scfg, step=0, run_id=label, optimizer=opt, reg_bins=data.reg_bins)
                m = {"K": K, "n_before": n, "n_cases": model.case_count(), **base_metrics, **test_metrics(model, data, "test_nofinetune"), "audit_mean_loss_pre": ainfo["mean_loss_pre"]}
                hist = []
                if proto["finetune_epochs"] > 0:
                    ft = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc, epochs=proto["finetune_epochs"], optimizer=opt)
                    hist = ft.history
                m.update(test_metrics(model, data))
                m.update(truth_retention(model, data))
                m.update({f"sel_{k}": v for k, v in res["summary"].items() if isinstance(v, (int, float))})
                self.finish(ds=ds, seed=seed, cond_label=label, cond={**cond, "K": K_spec}, data=data, model=model, store=store, archive=archive, metrics=m, history={"core": tr.history, "finetune": hist}, events=res["events"],
                            components={"retrieve": "trained NN-kNN core", "reuse_or_adapter": "off", "revise": "off", "retain": cond["policy"], "mcb": "on" if cc.mcb_enabled else "off", "component_synchronization": "off"},
                            budgets={"case_capacity": K, "core_epochs_run": tr.epochs_run, "finetune_epochs": proto["finetune_epochs"]},
                            maintenance={"policy": cond["policy"], "frequency_or_safe_checkpoint": "after core training (post-hoc), then fixed finetune", "thresholds_and_smoothing": self.cfg["statistics"], "archive_and_restore": "CaseArchive (exact state)"},
                            model_desc={"baseline_or_nnknn_variant": "t1 core", "core": cc.__dict__, "bias_init": getattr(base, "t1_bias_init", None)})

    def run_reuse(self, ds, seed):
        data = self.data_for(ds, seed)
        if data.task_type != "classification":
            raise ValueError("reuse experiment is the classification extension")
        model, cc, tr = self.trained_core(data, seed)
        store_a = CaseStatisticsStore()
        audit(model, data, self.cfg, store_a, 0)
        for cond in self.cfg["conditions"]:
            rc = ReuseConfig(seed=seed, **{**self.cfg["reuse"], **cond})
            adapter, info = train_classification_adapter(model, data, rc)
            ev = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc)
            # retention statistics must be pre-adaptation: re-audit with the adapter attached
            model.classification_adapter = adapter
            store_b = CaseStatisticsStore()
            audit(model, data, self.cfg, store_b, 0)
            del model.classification_adapter
            invariant = all(abs(store_a.get(c).positive_support - store_b.get(c).positive_support) < 1e-9 for c, _ in store_a.items())
            label = f"{cond['output_mode']}_{cond['loss']}"
            m = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)}, **{f"test_flip_{k}": v for k, v in ev["flips"].items()}, "retention_stats_pre_adaptation_invariant": invariant}
            self.finish(ds=ds, seed=seed, cond_label=label, cond=cond, data=data, model=model, store=store_a, archive=None, metrics=m, history={"core": tr.history, "adapter": info["history"]}, events=[],
                        components={"retrieve": "trained NN-kNN core (frozen)", "reuse_or_adapter": f"aggregate label-conditioned NN-CDH ({cond['output_mode']}, {cond['loss']})", "revise": "off", "retain": "none", "mcb": "off", "component_synchronization": "off (retrieval frozen first)"},
                        budgets={"case_capacity": model.case_count(), "adapter_epochs_max": rc.epochs}, maintenance={"policy": "none"}, model_desc={"core": cc.__dict__, "adapter": rc.__dict__})

    def run_sync(self, ds, seed):
        data = self.data_for(ds, seed)
        base, cc, tr = self.trained_core(data, seed)
        fr, _ = calibrate_free_radius(base, s_task=float(self.cfg["sync"]["s_task"]), snapshot_step=tr.best_epoch)
        from model.t1.geometry import case_case_distance

        D = case_case_distance(base)
        near_scale = float(D[torch.isfinite(D) & (D > 0)].median())
        for cond in self.cfg["conditions"]:
            model = copy.deepcopy(base)
            fields = SyncConfig.__dataclass_fields__
            merged = {**self.cfg["sync"], **cond}
            sc = SyncConfig(seed=seed, **{k: (tuple(v) if k == "hidden_dims" else v) for k, v in merged.items() if k in fields and k != "seed"})
            store, archive, events = CaseStatisticsStore(), CaseArchive(), []
            K = resolve_K(self.cfg["sync"].get("K", 1.0), model.case_count()) if "rrr" in sc.schedule else None

            def hook(ep, mdl, opt):
                audit(mdl, data, self.cfg, store, ep)
                res = run_maintenance(mdl, store, archive, K, retention_cfg_for({}, self.cfg, seed), score_cfg_for(self.cfg), step=ep, run_id=cond["schedule"], optimizer=opt, reg_bins=data.reg_bins)
                events.extend(res["events"])
                return res["summary"]

            recal = lambda ep: calibrate_free_radius(model, s_task=float(self.cfg["sync"]["s_task"]), snapshot_step=ep)[0]  # noqa: E731
            m_epochs = set(self.cfg["sync"].get("maintenance_epochs", []))
            adapter, info = train_synchronized(model, data, cc, sc, fr, near_scale=near_scale, maintenance_hook=hook, maintenance_epochs=m_epochs, recalibrate=recal, core_optimizer=clone_optimizer(tr.optimizer, model, cc))
            rc = ReuseConfig(output_mode=sc.output_mode, loss="combined", lambda_diff=1.0, lambda_cls=1.0, probability_mode=self.cfg["sync"].get("probability_mode", "softmax"), seed=seed)
            ev = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc)
            m = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)}, **{f"test_flip_{k}": v for k, v in ev["flips"].items()}, "tau_task": fr.tau_task, "free_radius_status": fr.status, "n_cases": model.case_count()}
            self.finish(ds=ds, seed=seed, cond_label=cond["schedule"], cond=cond, data=data, model=model, store=store if events else None, archive=archive, metrics=m, history={"core": tr.history, "sync": info["history"]}, events=events,
                        components={"retrieve": "NN-kNN core", "reuse_or_adapter": sc.output_mode, "revise": "off", "retain": "checkpoint maintenance" if events else "none", "mcb": "off", "component_synchronization": sc.schedule},
                        budgets={"case_capacity": model.case_count(), "sync_epochs_max": sc.epochs}, maintenance={"policy": self.cfg["retention"]["policy"] if events else "none"},
                        model_desc={"core": cc.__dict__, "sync": sc.__dict__, "free_radius_at_t_star": fr.to_dict()})

    def run_mcb(self, ds, seed):
        data = self.data_for(ds, seed)
        for cond in self.cfg["conditions"]:
            cc = core_cfg_for(self.cfg, data.task_type, seed, mcb_enabled=bool(cond["mcb"]))
            model = build_model(data.X_train, data.y_train, cc, data.num_classes)
            store, archive, events = CaseStatisticsStore(data.task_type), CaseArchive(), []
            tracker = StabilityTracker(data.X_val, k=cc.top_k)
            tracker.snapshot(model, 0)
            K = resolve_K(cond["K"], model.case_count())
            rcfg = retention_cfg_for(cond, self.cfg, seed)
            scfg = score_cfg_for(self.cfg)

            def hook(ep, mdl, opt):
                audit(mdl, data, self.cfg, store, ep)
                res = run_maintenance(mdl, store, archive, K, rcfg, scfg, step=ep, run_id=f"mcb{cond['mcb']}", optimizer=opt, reg_bins=data.reg_bins)
                events.extend(res["events"])
                return {**res["summary"], **tracker.snapshot(mdl, ep, res["kept_case_ids"])}

            tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc, checkpoint_hook=hook, checkpoint_epochs=set(self.cfg["maintenance_epochs"]))
            final = tracker.snapshot(model, tr.epochs_run + 1)
            recs = tracker.records
            m = {**test_metrics(model, data), "n_cases": model.case_count(), **truth_retention(model, data)}
            for key in ("representation_drift", "neighborhood_churn", "selection_churn"):
                vals = [r[key] for r in recs if key in r]
                if vals:
                    m[f"mean_{key}"] = float(np.mean(vals))
            label = f"mcb{int(bool(cond['mcb']))}_{cond['policy']}_K{cond['K']}"
            self.finish(ds=ds, seed=seed, cond_label=label, cond=cond, data=data, model=model, store=store, archive=archive, metrics=m, history={"core": tr.history, "stability": recs}, events=events,
                        components={"retrieve": "NN-kNN core with MLP encoder", "reuse_or_adapter": "off", "revise": "off", "retain": cond["policy"], "mcb": "on" if cond["mcb"] else "off", "component_synchronization": "off"},
                        budgets={"case_capacity": K, "core_epochs_run": tr.epochs_run}, maintenance={"policy": cond["policy"], "frequency_or_safe_checkpoint": f"epochs {self.cfg['maintenance_epochs']}"},
                        model_desc={"core": cc.__dict__})

    def run_revise(self, ds, seed):
        data = self.data_for(ds, seed)
        if not data.case_truth or "corrupted" not in data.case_truth:
            raise ValueError("revise experiment needs a synthetic dataset with known corruption")
        model, cc, tr = self.trained_core(data, seed)
        store = CaseStatisticsStore()
        audit(model, data, self.cfg, store, 0)
        ids = model.active_case_ids().cpu().numpy()
        scores = score_cases(ids, active_cohorts(model), model.biases[: model.case_count()].detach().cpu().numpy(), store, score_cfg_for(self.cfg))
        truth = data.case_truth["corrupted"]
        true_labels = data.meta.get("true_labels")
        if true_labels is None:
            raise ValueError("synthetic data must expose true labels for the simulated reviewer")
        rv = self.cfg["revise"]
        rows = flagging_ablation(model, data, scores, cc, truth_corrupted=truth, true_labels=np.asarray(true_labels), budgets=rv["budgets"], methods=rv["methods"], retrain_epochs=rv["retrain_epochs"], run_id=f"{self.exp}-s{seed}", seed=seed, core_optimizer=tr.optimizer)
        for r in rows:
            label = f"{r['method']}_b{r['review_budget']}"
            m = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for ck in ("M0", "M1", "M2"):
                for k, v in r[ck].items():
                    m[f"test_{ck}_{k}"] = v
            for fk in ("flips_M0_M1", "flips_M0_M2", "influenced_flips_M0_M1", "collateral_flips_M0_M1"):
                if fk in r:
                    for k, v in r[fk].items():
                        m[f"{fk}_{k}"] = v
            self.finish(ds=ds, seed=seed, cond_label=label, cond={"method": r["method"], "review_budget": r["review_budget"]}, data=data, model=model, store=store, archive=None, metrics=m, history={"core": tr.history}, events=[],
                        components={"retrieve": "NN-kNN core", "reuse_or_adapter": "off", "revise": "simulated oracle review (T1.2 harness)", "retain": "none", "mcb": "off", "component_synchronization": "off"},
                        budgets={"review_budget": r["review_budget"], "retrain_epochs": rv["retrain_epochs"]}, maintenance={"policy": "none"}, model_desc={"core": cc.__dict__}, extra_files={"interventions.json": r["interventions"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=str(REPO / "results" / "t1"))
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    runner = Runner(cfg, Path(args.out))
    fn = getattr(runner, f"run_{cfg['type']}")
    seeds = args.seeds if args.seeds else cfg["seeds"]
    for ds in cfg["datasets"]:
        if args.only_dataset and ds["name"] != args.only_dataset:
            continue
        for seed in seeds:
            t0 = time.time()
            fn(ds, int(seed))
            print(f"[timing] {cfg['experiment']} {ds['name']} s{seed}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
