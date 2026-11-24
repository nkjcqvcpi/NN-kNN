"""Run a T1 experiment from a YAML config; every run is recorded as a run unit.

    python tools/t1_run.py configs/t1/<config>.yaml --study S [--only-dataset iris] [--seeds 0 1]

Experiment types (``type:``): legacy_reference, retention, reuse, reuse_retention, encoder_ablation,
removal_retraining, adapted_retrained_removal, sync, mcb, revise.

``--study S`` is required. Every run is one unit ``NN-KNN/runs/t1/S/<experiment>/<dataset>/<condition>/s<seed>/``
(``model.t1.records``; ``$NNKNN_RUNS`` overrides the run root): records ``manifest``, ``metrics`` (and ``final/*``
numbers), ``history`` (and its curves), ``case_statistics``, ``case_maintenance``, ``retrieval_events``, ``row`` and
one record per extra JSON part; weights as named states (``checkpoint`` and the type's parts, e.g. ``M0``,
``initial``, ``starting_checkpoint``; legacy_reference also ``best_retrieval``/``best_adapter``/``final`` from
``train_model``); ``rng``, the RNG state the run left. No per-run file, shard row or checkpoint file is written.
``--out`` holds only the inputs: the source snapshot zip and ``<dataset>/data_snapshot_s<seed>.pt``.

Never retrained: training that several runs share (the trained core, a type's initial adapter, the retrained
removal of one K, the revise ablation) is recorded as a stage unit ``.../<dataset>/_<stage>/s<seed>/`` (named states
``objects``, the exact objects, and ``rng``). Before any training a rerun checks every unit: a complete run is
skipped and the RNG state it recorded is restored, a complete stage is loaded instead of trained, and a stage whose
runs are all complete is neither trained nor loaded. A partly rerun study therefore computes the same numbers and
weights as an uninterrupted one. A complete unit with another config raises ConfigMismatchError before training.

Values required by the plan that remain unresolved PI decisions must be given
explicitly in the config; ``null`` or ``"..."`` aborts the run.
``NNKNN_DETERMINISTIC=1`` enables ``torch.use_deterministic_algorithms`` for the run.
"""

from __future__ import annotations

import argparse
import copy
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

from model.common.artifacts import build_manifest, case_statistics_rows, config_id, source_fingerprint  # noqa: E402
from model.t1.calibration import calibrate_free_radius, compact_calibration_snapshot  # noqa: E402
from model.common.core import CoreConfig, build_model, clone_optimizer, evaluate, make_optimizer, retrieval_events, train_retrieval  # noqa: E402
from model.common.geometry import active_case_features, query_features  # noqa: E402
from model.t1.data import make_splits  # noqa: E402
from model.t1.candidates import MaintenanceReference
from model.t1.maintenance import CaseArchive, run_maintenance  # noqa: E402
from model.t1.mcb import StabilityTracker  # noqa: E402
from model.t1.provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, audit_provenance, score_cases  # noqa: E402
from model.t1.retention import RetentionConfig  # noqa: E402
from model.t1.reuse import ReuseConfig, calibrate_temperature, evaluate_reuse, evaluate_regression_reuse, reuse_logits, train_classification_adapter, train_regression_adapter  # noqa: E402
from model.t1.revise import flagging_ablation  # noqa: E402
from model.t1.sync import SyncConfig, train_synchronized  # noqa: E402
from model.t1 import records  # noqa: E402
from model.common.nominal import encoded_queries  # noqa: E402
from model.common.tracking import (ConfigMismatchError, config_identity, load_run, load_unit, open_run,  # noqa: E402
                                   resolve_unit, run_status)


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
    expand_condition_grid(cfg)
    return cfg


def expand_condition_grid(cfg: dict[str, Any]) -> None:
    """Append ``condition_grid`` blocks to ``conditions`` as named Cartesian products.

    Each block is ``{name, fixed: {...}, grid: {key: [values]}}``; a condition is
    named ``<name>__<key>=<value>...`` in grid-key order. Names must be unique.
    """
    import itertools
    blocks = cfg.pop("condition_grid", None)  # expanded once; saved configs list conditions
    if not blocks:
        return
    conds = list(cfg.get("conditions") or [])
    for block in blocks:
        keys = list(block.get("grid", {}))
        for values in itertools.product(*(block["grid"][k] for k in keys)):
            cond = {**block.get("fixed", {}), **dict(zip(keys, values))}
            cond["name"] = block["name"] + "".join(f"__{k}={v}" for k, v in zip(keys, values))
            conds.append(cond)
    names = [c.get("name", c.get("schedule")) for c in conds]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    cfg["conditions"] = conds


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
        min_retrieval_count=float(s["min_retrieval_count"]),
        min_activation_mass=float(s["min_activation_mass"]),
        bias_normalization=s.get("bias_normalization", "within_cohort_percentile"),
        bias_source=s.get("bias_source", "current"),
    )


def resolve_K(k: Any, n: int) -> int:
    if isinstance(k, float) and k <= 1.0:
        return max(1, int(round(k * n)))
    return int(k)


# --------------------------------------------------------------------------- helpers


def maintenance_reference(data, cfg, adapter=None):
    src = cfg["statistics"]["source"]
    tolerance = cfg["statistics"].get("regression_success_tolerance")
    if src == "train_loo":
        return MaintenanceReference(data.X_train, data.y_train, torch.arange(len(data.y_train)), tolerance, adapter, src, encoded_queries(adapter, data, "train"))
    if src == "maintenance_split" and data.X_maint is not None:
        return MaintenanceReference(data.X_maint, data.y_maint, None, tolerance, adapter, src, encoded_queries(adapter, data, "maint"))
    raise ValueError("Maintenance reference must be train_loo or a separate maintenance split")


def audit(model, data, cfg, store, step, adapter=None):
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
    ref = maintenance_reference(data, cfg, adapter)
    return audit_provenance(model, X, y, store, exclude_identical=excl,
                            counterfactual=bool(cfg["statistics"]["counterfactual"]),
                            step=step, reg_bins=data.reg_bins, query_case_ids=ref.query_case_ids,
                            adapter=adapter, regression_success_tolerance=ref.regression_success_tolerance, query_nominal=ref.query_nominal)


def retention_cfg_for(cond: dict[str, Any], cfg: dict[str, Any], seed: int) -> RetentionConfig:
    r = dict(cfg["retention"])
    r.update({k: v for k, v in cond.items() if k in RetentionConfig.__dataclass_fields__})
    r["seed"] = seed
    r.setdefault("notes", {})
    return RetentionConfig(**r)


# Sync condition keys besides SyncConfig fields: the label, the core start, the
# retrieval learning-rate scale and free-radius calibration (each merged over
# ``sync``), and in-training maintenance (``sync_maintenance_for``).
SYNC_RUNNER_KEYS = frozenset({"name", "core_pretrain", "retrieval_lr_scale", "s_task", "free_radius_method",
                              "radius_aggregation", "bias_transform", "shrink_factor", "clipping_quantile"})
SYNC_MAINTENANCE_KEYS = frozenset({"K", "maintenance_epochs"})


def sync_maintenance_for(cond: dict[str, Any], cfg: dict[str, Any], seed: int):
    """In-training maintenance of one sync condition: ``(retention, K spec, epochs)``.

    For this condition only, its RetentionConfig fields (``policy``,
    ``min_per_cohort``, ...) override ``retention``, ``K`` overrides the hook's
    capacity (``capacity_match.K`` if set, else ``sync.K``, else 1.0) and
    ``maintenance_epochs`` overrides ``sync.maintenance_epochs``, so one config
    can compare policies on an ``_rrr`` schedule. Only ``_rrr`` schedules maintain
    during training; on other schedules these keys would be inert yet recorded in
    the row, so they raise. The post-hoc capacity match stays one protocol for every
    condition. A key the runner does not read was once dropped silently; it raises.
    """
    retention_keys = set(RetentionConfig.__dataclass_fields__) - {"seed"}  # the run seed always wins
    allowed = (set(SyncConfig.__dataclass_fields__) - {"seed"}) | retention_keys | SYNC_RUNNER_KEYS | SYNC_MAINTENANCE_KEYS
    label = cond.get("name", cond.get("schedule"))
    unknown = sorted(set(cond) - allowed)
    if unknown:
        raise ValueError(f"sync condition {label!r}: unknown key(s) {unknown}; a condition takes SyncConfig fields, "
                         f"RetentionConfig fields, K, maintenance_epochs or {sorted(SYNC_RUNNER_KEYS)}")
    schedule = str({**cfg["sync"], **cond}.get("schedule", ""))
    own = set(cond) & (retention_keys | SYNC_MAINTENANCE_KEYS)
    if own and "rrr" not in schedule:
        raise ValueError(f"sync condition {label!r}: {sorted(own)} have no effect; schedule {schedule!r} "
                         f"does not maintain during training")
    capacity_match = cfg.get("capacity_match")
    K = cond.get("K", capacity_match["K"] if capacity_match else cfg["sync"].get("K", 1.0))
    epochs = set(cond.get("maintenance_epochs", cfg["sync"].get("maintenance_epochs", [])))
    if "rrr" not in schedule:
        return None, K, epochs
    retention = retention_cfg_for(cond, cfg, seed)
    retention.validate()  # a bad policy fails before any training, not at the first hook
    return retention, K, epochs


def test_metrics(model, data, prefix="test"):
    e = evaluate(model, data.X_test, data.y_test)
    out = {f"{prefix}_{k}": v for k, v in e.items() if not k.startswith(("pred", "prob"))}
    if data.test_groups and model.task_type == "classification":
        pred = e["pred_pre"]
        for g, m in data.test_groups.items():
            mt = torch.as_tensor(m)
            out[f"{prefix}_accuracy_{g}"] = float((pred[mt] == data.y_test[mt]).float().mean())
    return out


def val_metrics(model, data):
    """Retrieval-only validation metrics: the only stream allowed to rank conditions."""
    e = evaluate(model, data.X_val, data.y_val)
    return {f"val_{k}": v for k, v in e.items() if not k.startswith(("pred", "prob"))}


def adapted_metrics(model, adapter, data, stream, cfg, rc=None):
    """Pre/post-adaptation metrics on one evaluation stream (``test`` or ``val``).

    Classification also reports ``ece_post_T``/``nll_post_T`` under one temperature
    fitted on the validation stream's adapted outputs (``temperature``, reported
    with the validation stream); predictions are unchanged. When the validation
    NLL has no minimizer inside the temperature bounds (for example, no validation
    row is misranked), T = 1 and ``temperature_degenerate`` is true.
    """
    X, y = getattr(data, f"X_{stream}"), getattr(data, f"y_{stream}")
    if data.task_type == "classification":
        temperature, degenerate = calibrate_temperature(
            reuse_logits(model, adapter, data.X_val, rc, query_nominal=encoded_queries(adapter, data, "val")), data.y_val)
        out = evaluate_reuse(model, adapter, X, y, rc, query_nominal=encoded_queries(adapter, data, stream),
                             test_groups=data.test_groups if stream == "test" else None, temperature=temperature)
        if stream == "val":
            out["temperature"] = temperature
            out["temperature_degenerate"] = degenerate
    else:
        out = evaluate_regression_reuse(model, adapter, X, y,
                                        success_tolerance=cfg["statistics"]["regression_success_tolerance"])
    # Synchronization's post-adaptation task loss (cross-entropy or MSE), so frozen-core
    # reuse and every schedule share one selection criterion.
    from model.common.outcomes import frozen_evaluation
    from model.t1.sync import _forward_terms
    device = model.cases.device
    nominal = encoded_queries(adapter, data, stream)
    classification = data.task_type == "classification"
    with frozen_evaluation(model, adapter):
        T = _forward_terms(model, adapter, X.to(device), y.to(device), None,
                           output_mode=rc.output_mode if classification else "aggregate_regression",
                           near_scale=1.0, query_nominal=None if nominal is None else nominal.to(device),
                           score_scale=rc.score_scale if classification else 1.0)
    out["L_post"] = float(T["L_post"])
    return out


@torch.no_grad()
def input_locality(model, X, batch_size=256):
    """Fixed-metric locality: mean over queries of sum_i w_qi * ||x_q - x_i||_2.

    Input space (every stream is train-standardized), with the prediction's
    weights: eval-mode retrieval without leave-one-out (queries are not cases).
    """
    from model.common.outcomes import frozen_evaluation
    device, total = model.cases.device, 0.0
    with frozen_evaluation(model):
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            r = model.retrieve(xb, exclude_identical=False)
            d = torch.cdist(xb.flatten(1), model.cases[r["case_indices"]].flatten(1),
                            compute_mode="donot_use_mm_for_euclid_dist")
            # accumulate in float64 on the CPU: some accelerators (Arc A770) lack fp64
            total += float((r["weights"].detach().cpu().double() * d.cpu().double()).sum())
    return total / len(X)


def to_cpu(obj):
    """``obj`` with every tensor in nested dicts, lists and tuples moved to the CPU.

    Containers without device tensors are returned unchanged, so CPU runs save
    exactly what they saved before; a rebuilt state dict keeps its ``_metadata``.
    """
    if isinstance(obj, torch.Tensor):
        return obj if obj.device.type == "cpu" else obj.detach().cpu()
    if isinstance(obj, dict):
        moved = {k: to_cpu(v) for k, v in obj.items()}
        if all(moved[k] is v for k, v in obj.items()):
            return obj
        out = copy.copy(obj)
        out.update(moved)
        return out
    if isinstance(obj, (list, tuple)):
        moved = [to_cpu(v) for v in obj]
        if all(m is v for m, v in zip(moved, obj)):
            return obj
        if isinstance(obj, list):
            return moved
        return type(obj)(*moved) if hasattr(obj, "_fields") else type(obj)(moved)
    return obj


class EveryEpoch:
    """Observation epochs for ``train_retrieval``: every epoch is a member, none is pending.

    ``train_retrieval`` runs its checkpoint hook at member epochs, and delays early
    stopping only for epochs it can iterate as still to come. A read-only observer
    must never delay stopping, so iteration is empty.
    """

    def __contains__(self, epoch) -> bool:
        return True

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return True


def stability_summary(records, prefix):
    out = {}
    for key in ("representation_drift", "neighborhood_churn"):
        values = [r[key] for r in records if key in r]
        if values:
            out[f"{prefix}_mean_{key}"] = float(np.mean(values))
            out[f"{prefix}_late_{key}"] = float(np.mean(values[-10:]))
    return out


@torch.no_grad()
def online_ema_gap(model):
    """Mean distance between online (query) and EMA (case) embeddings of the same stored cases."""
    if getattr(model, "momentum_encoder", None) is None:
        return {}
    cases = active_case_features(model)
    online = query_features(model, model.cases[: model.case_count()])
    return {"online_ema_gap": float((online - cases).norm(dim=1).mean()),
            "case_embedding_norm": float(cases.norm(dim=1).mean())}


def arm_label(arm, label):
    return label if arm is None else f"{arm['name']}__{label}"


def arm_fields(arm):
    return {} if arm is None else {"arm": arm["name"], **{k: v for k, v in arm.items() if k != "name"}}


def arm_core(arm):
    return {} if arm is None else {k: v for k, v in arm.items() if k in CoreConfig.__dataclass_fields__}


def near_scale_of(model):
    from model.common.geometry import case_case_distance
    D = case_case_distance(model)
    positive = D[torch.isfinite(D) & (D > 0)]
    return float(positive.median()) if positive.numel() else 1.0


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
    for key in ("corrupted", "duplicate", "rare"):
        if key in data.case_truth:
            clean &= ~data.case_truth[key]
    out["kept_fraction_clean_base"] = float(np.mean([i in kept for i in np.nonzero(clean)[0]]))
    return out


class RunSink:
    """One run's outputs recorded into its open run unit; see the module docstring."""

    RECORD = {"run_manifest.json": "manifest", "history.json": "history"}

    def __init__(self, run: Any) -> None:
        self.run = run

    def json(self, name: str, obj: Any) -> None:
        if name == "metrics.json":
            self.run.log_record("metrics", records.ordered(obj))
            self.run.log_final(records.numeric(obj))
            return
        self.run.log_record(self.RECORD.get(name, name.removesuffix(".json")), records.ordered(obj))
        if name == "history.json":
            records.log_curves(self.run, obj)

    def jsonl(self, name: str, rows: list) -> None:
        self.run.log_records(name.removesuffix(".jsonl"), [records.ordered(r) for r in rows])

    def torch(self, name: str, obj: Any) -> None:
        self.run.save_state(name.removesuffix(".pt"), records.storable(to_cpu(obj)))

    def row(self, row: dict) -> None:
        self.run.log_record("row", records.ordered(row))


class Runner:
    def __init__(self, cfg: dict[str, Any], out_root: Path, study: str) -> None:
        """``study``: every run is recorded as a unit of this study (required); ``out_root`` keeps the inputs only."""
        if not isinstance(study, str) or not study:
            raise TypeError("Runner needs a study: every T1 run is recorded as a run unit (tools/t1_run.py --study)")
        self.cfg = cfg
        self.study = study
        self.exp = cfg["experiment"]
        self.out = out_root / self.exp
        self.out.mkdir(parents=True, exist_ok=True)
        self.source_hash = source_fingerprint(REPO)
        self.source_path = self.out / f"source_snapshot_{self.source_hash[:12]}_{os.getpid()}.zip"
        import zipfile
        with zipfile.ZipFile(self.source_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for directory in ("model", "tools", "configs", "tests", "datasets"):
                for path in sorted((REPO / directory).rglob("*")):
                    if path.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in path.parts:
                        bundle.write(path, path.relative_to(REPO).as_posix())
        torch.set_num_threads(int(cfg.get("torch_threads", 1)))

    def data_for(self, ds: dict[str, Any], seed: int):
        sp = self.cfg["splits"]
        # A per-dataset cap wins; otherwise splits.max_train applies, as in tools/t1_grid_search.py.
        return make_splits(ds["name"], ds["task_type"], seed, test_frac=sp["test_frac"], val_frac=sp["val_frac"], maint_frac=sp.get("maint_frac", 0.0), reg_cohort_bins=sp.get("reg_cohort_bins", 5), max_train=ds.get("max_train", sp.get("max_train")), synthetic=ds.get("synthetic"))

    def dataset_key(self, ds):
        """Folder/row name of a task; role-qualified when the name has several task types here,
        or when the entry says ``qualified: true`` (one role per config, e.g. regression wine)."""
        roles = {d["task_type"] for d in self.cfg["datasets"] if d["name"] == ds["name"]}
        return f"{ds['task_type']}__{ds['name']}" if len(roles) > 1 or ds.get("qualified") else ds["name"]

    def core_config(self, data, seed: int, **over) -> CoreConfig:
        selected = self.cfg.get("core_by_dataset", {}).get(f"{data.task_type}__{data.name}", {})
        return core_cfg_for(self.cfg, data.task_type, seed, **{**selected, **over})

    def trained_core(self, data, seed: int, **over):
        cc = self.core_config(data, seed, **over)
        model = build_model(data.X_train, data.y_train, cc, data.num_classes)
        tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc)
        return model, cc, tr

    def observed_core(self, data, seed: int, **over):
        """``trained_core`` plus read-only per-epoch stability snapshots on validation probes."""
        cc = self.core_config(data, seed, **over)
        model = build_model(data.X_train, data.y_train, cc, data.num_classes)
        tracker = StabilityTracker(data.X_val, k=cc.top_k)
        tracker.snapshot(model, 0)
        tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc,
                             checkpoint_hook=lambda ep, mdl, opt: tracker.snapshot(mdl, ep),
                             checkpoint_epochs=EveryEpoch())
        return model, cc, tr, tracker

    def arms(self):
        return self.cfg.get("arms") or [None]

    def unit_key(self, ds, cond_label, seed):
        return records.key(self.study, self.exp, self.dataset_key(ds), cond_label, f"s{seed}")

    def run_config(self, ds, seed, cond):
        return {"experiment": self.exp, "dataset": ds, "seed": seed, "condition": cond, "config": self.cfg}

    @staticmethod
    def unit_config(run_cfg):
        """The config a unit is opened and checked with: ``run_cfg`` without the config's ``seeds`` list.

        ``--seeds`` rewrites that list (tools/t1_parallel.py and tools/t1_execute_study.py start one job per seed,
        t1_launch.ps1 and a plain t1_run none), and the unit's own seed is already ``run_cfg["seed"]``; the manifest,
        the row and the run id keep the full ``run_cfg``.
        """
        return records.jsonable({**run_cfg, "config": {k: v for k, v in run_cfg["config"].items() if k != "seeds"}})

    def check_recorded(self, ds, seed) -> None:
        """Before any training of ``(ds, seed)``: each complete unit of this experiment, dataset and seed must hold
        the config this run would give it (its condition aside), else ConfigMismatchError now, not after training."""
        expected = self.unit_config(self.run_config(ds, seed, {}))
        expected.pop("condition")
        expected = config_identity(expected)
        tree = records.unit_tree(self.study, self.exp, self.dataset_key(ds))
        for path in sorted(tree.glob(f"*/s{seed}")) if tree.is_dir() else ():
            info = run_status(path)
            if info.status != "complete":
                continue
            stored = dict(load_run(path, arrays=False).config or {})
            stored.pop("condition", None)
            if config_identity(stored) != expected:  # provenance keys aside (tracking.PROVENANCE_KEYS)
                raise ConfigMismatchError(f"{info.run_id} is complete with a different config (checked before training)")

    def recorded(self, key, run_cfg) -> bool:
        """True (and say so) when the unit is complete with this config; ConfigMismatchError for another config."""
        info = resolve_unit(key, config=self.unit_config(run_cfg))
        if info is not None:
            print(f"[recorded] {key.run_id} is complete; not written again", flush=True)
        return info is not None

    # ----------------------------------------------------------------- never retrained

    def complete(self, ds, seed, label, cond):
        """RunData of the unit ``label`` when it is complete with this run's config, else None; ConfigMismatchError
        when it is complete with another config."""
        return load_unit(self.unit_key(ds, label, seed), config=self.unit_config(self.run_config(ds, seed, cond)),
                         arrays=False)

    def skip(self, ds, seed, label, cond) -> bool:
        """Call before any training of one run: True when its unit is complete. The RNG state the run left is then
        restored, so the runs after it compute what they computed in an uninterrupted study."""
        data = self.complete(ds, seed, label, cond)
        if data is None:
            return False
        records.restore_rng_of(data)
        print(f"[recorded] {data.info.run_id} is complete; not trained again", flush=True)
        return True

    def stage(self, ds, seed, name, units, compute):
        """Shared training that the runs ``units`` (``[(label, cond)]``) continue, e.g. the trained core.

        Returns None, training and loading nothing, when every one of those runs is complete. Otherwise the stage
        unit ``<experiment>/<dataset>/_<name>/s<seed>`` provides it (``records.stage_unit``): loaded when complete
        (its objects, then the RNG state its training left), else ``compute()`` runs and its dict is recorded first.
        """
        if all(self.complete(ds, seed, label, cond) is not None for label, cond in units):
            return None
        key = self.unit_key(ds, f"_{name}", seed)

        def describe(run, objects):
            if "tr" in objects:
                run.log_record("history", records.ordered({"core": objects["tr"].history}))
                records.log_curves(run, {"core": objects["tr"].history})
        objects, loaded = records.stage_unit(key, self.unit_config(self.run_config(ds, seed, {"stage": name})),
                                             compute, describe)
        print(f"[recorded] {key.run_id} is complete; loaded, not trained again" if loaded else
              f"[stage] {key.run_id} recorded", flush=True)
        return objects

    def core_stage(self, ds, seed, data, units, arm=None, observed=False, extra=None):
        """The trained core of ``(ds, seed, arm)`` (``model``, ``cc``, ``tr`` and, observed, ``tracker``) as a stage;
        ``extra(objects)`` adds more shared training to the same stage (e.g. the initial adapter)."""
        def compute():
            over = arm_core(arm)
            if observed:
                model, cc, tr, tracker = self.observed_core(data, seed, **over)
                objects = {"model": model, "cc": cc, "tr": tr, "tracker": tracker}
            else:
                model, cc, tr = self.trained_core(data, seed, **over)
                objects = {"model": model, "cc": cc, "tr": tr}
            if extra is not None:
                objects.update(extra(objects))
            return objects
        return self.stage(ds, seed, "core" if arm is None else f"core__{arm['name']}", units, compute)

    def reuse_unit(self, task_type, cond, arm=None):
        output_mode = cond["output_mode"] if task_type == "classification" else "aggregate_regression"
        return arm_label(arm, cond.get("name", f"{output_mode}_{cond['loss']}")), {**arm_fields(arm), **cond}

    def sync_unit(self, cond, arm=None):
        capacity_match = self.cfg.get("capacity_match")
        capacity_spec = capacity_match["K"] if capacity_match else self.cfg["sync"].get("K", 1.0)
        label = cond.get("name", cond["schedule"]) + (f"_matched_K{capacity_spec}" if capacity_match else "")
        return arm_label(arm, label), {**arm_fields(arm), **cond}

    def finish(self, *, ds, seed, cond_label, cond, run=None, **kw):
        """Record one run into its unit (opened here unless ``run`` is given) with the RNG state and ID counter
        positions it leaves (named states ``rng``, ``ids``); returns them."""
        if run is not None:
            self._finish(RunSink(run), ds=ds, seed=seed, cond_label=cond_label, cond=cond, **kw)
            return records.save_rng(run)
        key, run_cfg = self.unit_key(ds, cond_label, seed), self.run_config(ds, seed, cond)
        if self.recorded(key, run_cfg):  # skip() refuses this before training; kept for a concurrent writer
            return None
        with open_run(key, config=self.unit_config(run_cfg)) as unit:
            self._finish(RunSink(unit), ds=ds, seed=seed, cond_label=cond_label, cond=cond, **kw)
            saved = records.save_rng(unit)
        records.restore_after(saved)  # closing the unit leaves the RNG and the ID counters as the run did
        return saved

    def _finish(self, sink, *, ds, seed, cond_label, cond, data, model, store, archive, metrics, history, events, components, budgets, maintenance, model_desc, extra_files=None, prediction_adapter=None, binary_files=None, optimizer=None, adapter_optimizer_state=None, optimizer_checkpoint_epoch=None):
        dataset_key = self.dataset_key(ds)
        run_cfg = self.run_config(ds, seed, cond)
        run_id = f"{self.exp}-{dataset_key}-{cond_label}-s{seed}-{config_id(run_cfg)[:8]}"
        man = build_manifest(
            run_id=run_id, repo=REPO, cfg=run_cfg, data_desc=data.describe(), components=components, budgets=budgets,
            maintenance=maintenance, evaluation={"confirmatory_or_exploratory": self.cfg["status"], "number_of_seeds": len(self.cfg["seeds"]), "uncertainty_method": "per-seed runs; summarized by tools/t1_summarize.py"}, model_desc=model_desc,
        )
        # provenance only (owner's order 2026-11-23: units are compared by configuration, not by code)
        man["run"]["source_changed_during_run"] = man["run"]["source_fingerprint_sha256"] != self.source_hash
        man["run"]["source_fingerprint_sha256"] = self.source_hash
        man["run"]["source_snapshot"] = str(self.source_path)
        man["run"]["device"] = str(model.cases.device)
        man["run"]["deterministic"] = torch.are_deterministic_algorithms_enabled()
        sink.json("run_manifest.json", man)
        sink.json("metrics.json", metrics)
        sink.json("history.json", history)
        sink.jsonl("case_statistics.jsonl", [] if store is None else case_statistics_rows(model, store, archive or CaseArchive()))
        sink.jsonl("case_maintenance.jsonl", events or [])
        sink.jsonl("retrieval_events.jsonl", retrieval_events(model, data.X_test, data.y_test, stream="test", run_id=run_id, adapter=prediction_adapter, query_nominal=encoded_queries(prediction_adapter, data, "test")))
        # CPU tensors only, so accelerator runs reload anywhere (a no-op on CPU).
        sink.torch("checkpoint.pt", {"model_state": model.state_dict(), "active_case_count": model.case_count(),
                    "case_statistics": None if store is None else store.state_dict(),
                    "archive": None if archive is None else archive.state_dict(),
                    "adapter_state": None if prediction_adapter is None else prediction_adapter.state_dict(),
                    "optimizer_state": None if optimizer is None else optimizer.state_dict(),
                    "adapter_optimizer_state": adapter_optimizer_state,
                    "optimizer_checkpoint_epoch": optimizer_checkpoint_epoch,
                    "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
                    "adapter_requires_grad": None if prediction_adapter is None else
                        {n: p.requires_grad for n, p in prediction_adapter.named_parameters()},
                    "model_description": model_desc})
        data_path = self.out / dataset_key / f"data_snapshot_s{seed}.pt"
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)  # recorded runs create no run directory
            snapshot = {name: getattr(data, name) for name in
                        ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                         "X_maint", "y_maint", "y_scale", "meta", "nominal_train", "nominal_val", "nominal_test", "nominal_maint")}
            snapshot.update({name: {k: torch.as_tensor(v) for k, v in getattr(data, name).items()}
                             for name in ("case_truth", "test_groups")})
            torch.save(to_cpu(snapshot), data_path)
        for name, obj in (extra_files or {}).items():
            sink.json(name, obj)
        if getattr(prediction_adapter, "nominal_schema", None) is not None:
            sink.json("adapter_nominal_schema.json", prediction_adapter.nominal_schema.state_dict())
        for name, obj in (binary_files or {}).items():
            sink.torch(name, obj)
        row = {"run_id": run_id, "experiment": self.exp, "dataset": dataset_key, "dataset_name": ds["name"], "task_type": data.task_type, "seed": seed, "condition": cond_label, **{k: v for k, v in cond.items() if not isinstance(v, (dict, list))}, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        sink.row(row)
        keys = ("test_accuracy_pre", "test_accuracy_post", "test_rmse_pre", "test_accuracy_rare", "n_cases")
        print(f"[done] {run_id} " + " ".join(f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics.get(k), (int, float))), flush=True)

    # ----------------------------------------------------------------- experiment types

    def run_legacy_reference(self, ds, seed):
        """Phase 0: the maintained NN-kNN trainer (current behavior) on the same splits."""
        cond = {"variant": "maintained_train_model"}
        if self.skip(ds, seed, "legacy", cond):  # before the data is loaded
            return None
        key, run_cfg = self.unit_key(ds, "legacy", seed), self.run_config(ds, seed, cond)
        data = self.data_for(ds, seed)
        if data.task_type == "classification":
            from model.classification_workflow import make_classification_cfg

            lcfg = make_classification_cfg(self.cfg["legacy"].get("overrides", {}))
        else:
            from model.regression_workflow import make_regression_cfg

            lcfg = make_regression_cfg({**self.cfg["legacy"].get("overrides", {}), **self.cfg["legacy"]["regression_overrides"]})
        # train_model writes no checkpoint file: it records its curves and best/final states into the unit and
        # resumes an interrupted training from the unit's resume checkpoint
        with open_run(key, config=self.unit_config(run_cfg)) as run:
            saved = self._legacy_reference(ds, seed, data, lcfg, cond, run)
        records.restore_after(saved)

    def _legacy_reference(self, ds, seed, data, lcfg, cond, run):
        from model.nnknn_model import train_model

        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        _, _, model = train_model(data.X_train, data.y_train, data.X_val, data.y_val, None, lcfg, run=run, resumable=True)
        m = test_metrics(model, data)
        m["train_seconds"] = time.time() - t0
        m["n_cases"] = model.case_count()
        return self.finish(ds=ds, seed=seed, cond_label="legacy", cond=cond, run=run, data=data, model=model, store=None, archive=None, metrics=m, history=[], events=[],
                    components={"retrieve": "maintained core", "reuse_or_adapter": "off", "revise": "off", "retain": "none (full memory)", "mcb": "off", "component_synchronization": "off"},
                    budgets={"case_capacity": model.case_count(), "training_epochs_max": lcfg.get("training_epochs")}, maintenance={"policy": "none"}, model_desc={"baseline_or_nnknn_variant": "legacy maintained train_model", "cfg": lcfg})

    def run_retention(self, ds, seed):
        data = self.data_for(ds, seed)
        units = [(f"{cond['policy']}_K{K_spec}", {**cond, "K": K_spec}) for cond in self.cfg["conditions"] for K_spec in self.cfg["K"]]
        core = self.core_stage(ds, seed, data, units)
        if core is not None:
            base, cc, tr = core["model"], core["cc"], core["tr"]
            base_metrics = test_metrics(base, data, "fullmem")
        proto = self.cfg["protocol"]
        for cond in self.cfg["conditions"]:
            for K_spec in self.cfg["K"]:
                if self.skip(ds, seed, f"{cond['policy']}_K{K_spec}", {**cond, "K": K_spec}):
                    continue
                model = copy.deepcopy(base)
                n = model.case_count()
                K = resolve_K(K_spec, n)
                label = f"{cond['policy']}_K{K_spec}"
                store, archive = CaseStatisticsStore(data.task_type), CaseArchive()
                rcfg = retention_cfg_for(cond, self.cfg, seed)
                scfg = score_cfg_for(self.cfg)
                opt = clone_optimizer(tr.optimizer, model, cc)  # continue the core's Adam state (see clone_optimizer)
                ainfo = audit(model, data, self.cfg, store, step=0)
                res = run_maintenance(model, store, archive, K, rcfg, scfg, step=0, run_id=label, optimizer=opt, reg_bins=data.reg_bins, reference=maintenance_reference(data, self.cfg))
                m = {"K": K, "n_before": n, "n_cases": model.case_count(), **base_metrics, **test_metrics(model, data, "test_nofinetune"), "audit_mean_loss_pre": ainfo["mean_loss_pre"], "audit_mean_loss_final": ainfo["mean_loss_final"]}
                hist = []
                if proto["finetune_epochs"] > 0:
                    ft = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc, epochs=proto["finetune_epochs"], optimizer=opt, lr_scale=float(proto["finetune_lr_scale"]), include_initial=True)
                    hist = ft.history
                m.update(test_metrics(model, data))
                m.update(truth_retention(model, data))
                m.update({f"sel_{k}": v for k, v in res["summary"].items() if isinstance(v, (int, float))})
                self.finish(ds=ds, seed=seed, cond_label=label, cond={**cond, "K": K_spec}, data=data, model=model, store=store, archive=archive, metrics=m, history={"core": tr.history, "finetune": hist}, events=res["events"],
                            components={"retrieve": "trained NN-kNN core", "reuse_or_adapter": "off", "revise": "off", "retain": cond["policy"], "mcb": "on" if cc.mcb_enabled else "off", "component_synchronization": "off"},
                            budgets={"case_capacity": K, "core_epochs_run": tr.epochs_run, "finetune_epochs": proto["finetune_epochs"]},
                            maintenance={"policy": cond["policy"], "frequency_or_safe_checkpoint": "after core training (post-hoc), then fixed finetune", "thresholds_and_smoothing": self.cfg["statistics"], "archive_and_restore": "CaseArchive (exact state)"},
                            model_desc={"baseline_or_nnknn_variant": "t1 core", "core": cc.__dict__, "bias_init": getattr(base, "t1_bias_init", None)}, extra_files={"maintenance_scoring_trace.json": res.get("scoring_trace", []), "maintenance_summary.json": res["summary"]})

    def run_reuse(self, ds, seed):
        data = self.data_for(ds, seed)
        for arm in self.arms():
            units = [self.reuse_unit(data.task_type, cond, arm) for cond in self.cfg["conditions"]]
            core = self.core_stage(ds, seed, data, units, arm)
            if core is not None:
                store_a = CaseStatisticsStore(data.task_type)
                audit(core["model"], data, self.cfg, store_a, 0)
            for cond in self.cfg["conditions"]:
                if not self.skip(ds, seed, *self.reuse_unit(data.task_type, cond, arm)):
                    self.reuse_condition(ds, seed, data, core["model"], core["cc"], core["tr"], cond, arm)

    def reuse_condition(self, ds, seed, data, model, cc, tr, cond, arm=None):
        """Train one frozen-retrieval adapter on ``model`` and record pre/post outcomes.

        An optional ``name`` labels the condition (default ``<output_mode>_<loss>``);
        every other key is a ``ReuseConfig`` field.
        """
        rc = ReuseConfig(seed=seed, **{**self.cfg["reuse"], **{k: v for k, v in cond.items() if k != "name"}})
        if data.task_type == "classification":
            adapter, info = train_classification_adapter(model, data, rc)
            output_mode = cond["output_mode"]
        else:
            adapter, info = train_regression_adapter(model, data, rc)
            output_mode = "aggregate_regression"
        ev = adapted_metrics(model, adapter, data, "test", self.cfg, rc)
        vev = adapted_metrics(model, adapter, data, "val", self.cfg, rc)
        store_b = CaseStatisticsStore(data.task_type)
        ainfo = audit(model, data, self.cfg, store_b, 0, adapter=adapter)
        label = self.reuse_unit(data.task_type, cond, arm)[0]
        m = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)},
             **{f"test_flip_{k}": v for k, v in ev.get("flips", {}).items()},
             **{f"val_{k}": v for k, v in vev.items() if not isinstance(v, dict)},
             "audit_mean_loss_final": ainfo["mean_loss_final"],
             "provenance_semantics": ainfo["provenance_semantics"],
             "test_locality_input": input_locality(model, data.X_test)}
        self.finish(ds=ds, seed=seed, cond_label=label, cond={**arm_fields(arm), **cond}, data=data, model=model, store=store_b, archive=None, metrics=m, history={"core": tr.history, "adapter": info["history"]}, events=[],
                    components={"retrieve": "trained NN-kNN core (frozen)", "reuse_or_adapter": f"aggregate label-conditioned NN-CDH ({output_mode}, {cond['loss']})", "revise": "off", "retain": "none", "mcb": "on" if cc.mcb_enabled else "off", "component_synchronization": "off (retrieval frozen first)"},
                    budgets={"case_capacity": model.case_count(), "adapter_epochs_max": rc.epochs,
                             "adapter_epochs_run": info["epochs_run"], "adapter_best_epoch": info["best_epoch"]},
                    maintenance={"policy": "none"}, model_desc={"core": cc.__dict__, "adapter": rc.__dict__},
                    prediction_adapter=adapter, optimizer=tr.optimizer,
                    adapter_optimizer_state=info["optimizer_state"],
                    optimizer_checkpoint_epoch={"retrieval": tr.best_epoch, "adapter": info["best_epoch"]})

    def run_encoder_ablation(self, ds, seed):
        """Is the EMA case encoder (MCB) useful before any case-base maintenance?

        Each arm (e.g. raw features, one shared MLP encoder, an online query MLP
        with an EMA case encoder) trains one full-memory retrieval core with
        read-only per-epoch stability snapshots. Setting 1 records that core's
        retrieval-only output. Setting 2 adds reuse to the same core: frozen-
        retrieval adapters (``reuse_conditions``) and synchronized schedules
        (``conditions``). Arms share splits, seeds, budgets and retrieval rules.
        """
        for cond in self.cfg.get("conditions") or []:
            sync_maintenance_for(cond, self.cfg, seed)  # refuse a bad condition before any training
        data = self.data_for(ds, seed)
        for arm in self.cfg["arms"]:
            retrieval_only = (arm_label(arm, "retrieval_only"), {**arm_fields(arm), "setting": "retrieval_only"})
            units = [retrieval_only, *(self.reuse_unit(data.task_type, cond, arm) for cond in self.cfg.get("reuse_conditions", [])),
                     *(self.sync_unit(cond, arm) for cond in self.cfg.get("conditions") or [])]
            core = self.core_stage(ds, seed, data, units, arm, observed=True)
            if core is not None:
                base, cc, tr, tracker = core["model"], core["cc"], core["tr"], core["tracker"]
            if not self.skip(ds, seed, *retrieval_only):
                self.retrieval_only(ds, seed, data, base, cc, tr, tracker, arm)
            for cond in self.cfg.get("reuse_conditions", []):
                if not self.skip(ds, seed, *self.reuse_unit(data.task_type, cond, arm)):
                    self.reuse_condition(ds, seed, data, base, cc, tr, cond, arm)
            if self.cfg.get("conditions"):
                near_scale = None if core is None else near_scale_of(base)
                for cond in self.cfg["conditions"]:
                    if not self.skip(ds, seed, *self.sync_unit(cond, arm)):
                        self.sync_condition(ds, seed, data, base, cc, tr, cond, arm, near_scale)

    def retrieval_only(self, ds, seed, data, base, cc, tr, tracker, arm):
        """Setting 1 of the encoder ablation: the trained core's retrieval-only output."""
        m = {**test_metrics(base, data), **val_metrics(base, data), "n_cases": base.case_count(),
             "core_epochs_run": tr.epochs_run, "core_best_epoch": tr.best_epoch,
             **stability_summary(tracker.records, "core"), **online_ema_gap(base),
             "test_locality_input": input_locality(base, data.X_test)}
        self.finish(ds=ds, seed=seed, cond_label=arm_label(arm, "retrieval_only"),
                    cond={**arm_fields(arm), "setting": "retrieval_only"}, data=data, model=base,
                    store=None, archive=None, metrics=m, history={"core": tr.history}, events=[],
                    components={"retrieve": f"NN-kNN core ({cc.representation})", "reuse_or_adapter": "off", "revise": "off",
                                "retain": "none (full memory)", "mcb": "on" if cc.mcb_enabled else "off",
                                "component_synchronization": "off"},
                    budgets={"case_capacity": base.case_count(), "core_epochs_max": cc.epochs,
                             "core_epochs_run": tr.epochs_run},
                    maintenance={"policy": "none"}, model_desc={"core": cc.__dict__, "arm": arm},
                    optimizer=tr.optimizer, optimizer_checkpoint_epoch={"retrieval": tr.best_epoch})

    def run_adapted_retrained_removal(self, ds, seed):
        """Optional high-cost final-output removal with actual continued adapters."""
        from model.t1.adapted_retraining import continued_adapted_trial, run_adapted_retrained_removal, matched_random_continuation
        from model.t1.retraining import reference_loss
        data = self.data_for(ds, seed)
        rc = ReuseConfig(seed=seed, **self.cfg["reuse"])
        from model.t1.reuse import train_regression_adapter, evaluate_regression_reuse

        def initial(core):  # the initial adapter is shared training too: part of the core's stage
            adapter, info = (train_classification_adapter(core["model"], data, rc) if data.task_type == "classification"
                             else train_regression_adapter(core["model"], data, rc))
            return {"adapter": adapter, "info": info}
        units = [(f"{cond['scope']}_K{kspec}", cond) for cond in self.cfg["conditions"] for kspec in self.cfg["K"]]
        core = self.core_stage(ds, seed, data, units, extra=initial)
        if core is not None:
            base, cc, tr, initial_adapter, info = core["model"], core["cc"], core["tr"], core["adapter"], core["info"]
        actual_mode = rc.output_mode if data.task_type == "classification" else "aggregate_regression_residual"
        model_desc = None if core is None else {"core": cc.__dict__, "adapter": rc.__dict__, "actual_adapter_mode": actual_mode}
        def evaluate_adapted(m, a):
            if data.task_type == "classification":
                return evaluate_reuse(m, a, data.X_test, data.y_test, rc,
                    query_nominal=encoded_queries(a, data, "test"), test_groups=data.test_groups)
            return evaluate_regression_reuse(m, a, data.X_test, data.y_test,
                success_tolerance=self.cfg["statistics"]["regression_success_tolerance"])
        def checkpoint(m, a, opt, astate, *, initial=False):
            return {"model_state": m.state_dict(), "adapter_state": a.state_dict(),
                "optimizer_state": opt.state_dict(), "adapter_optimizer_state": astate,
                "active_case_count": m.case_count(), "model_description": model_desc,
                "requires_grad": {n: p.requires_grad for n, p in m.named_parameters()},
                "adapter_requires_grad": {n: p.requires_grad for n, p in a.named_parameters()},
                "optimizer_checkpoint_epoch": {"retrieval": tr.best_epoch if initial else "fixed_final",
                    "adapter": info["best_epoch"] if initial else "fixed_final"}}
        for cond in self.cfg["conditions"]:
            re, ae = cond["retrieval_epochs"], cond["adapter_epochs"]
            condition_cfg = copy.deepcopy(self.cfg)
            reference_stream = cond.get("reference_stream", self.cfg["statistics"]["source"])
            condition_cfg["statistics"]["source"] = reference_stream
            # Validate a real independent stream before training any forks.
            if core is not None:
                maintenance_reference(data, condition_cfg, initial_adapter).validate()
            for kspec in self.cfg["K"]:
                label = f"{cond['scope']}_K{kspec}"
                if self.skip(ds, seed, label, cond):
                    continue
                store = CaseStatisticsStore(data.task_type)
                audit(base, data, condition_cfg, store, 0, adapter=initial_adapter)
                model, opt, ad, ast, archive, result = run_adapted_retrained_removal(
                    base, tr.optimizer, initial_adapter, info["optimizer_state"], data, cc, rc,
                    maintenance_reference(data, condition_cfg, initial_adapter), store, score_cfg_for(condition_cfg),
                    retention_cfg_for(cond, self.cfg, seed), resolve_K(kspec, base.case_count()),
                    retrieval_epochs=re, adapter_epochs=ae, lr_scale=self.cfg["continuation"]["lr_scale"], run_id=label)
                # An unchanged full-memory trajectory gets the same accepted
                # number of phase blocks, minibatches and examples, not cost.
                control, cad, cast = copy.deepcopy(base), copy.deepcopy(initial_adapter), copy.deepcopy(info["optimizer_state"])
                copt = clone_optimizer(tr.optimizer, control, cc)
                control_history = []
                for _ in result["events"]:
                    control, copt, cad, caopt, ci = continued_adapted_trial(control, copt, cad, cast,
                        data, cc, rc, retrieval_epochs=re, adapter_epochs=ae,
                        lr_scale=self.cfg["continuation"]["lr_scale"])
                    cast = copy.deepcopy(caopt.state_dict())
                    control_history.append(ci)
                ev = evaluate_adapted(model, ad)
                cev = evaluate_adapted(control, cad)
                random_model, random_opt, random_ad, random_ast, random_info = matched_random_continuation(
                    base, tr.optimizer, initial_adapter, info["optimizer_state"], data, cc, rc, store,
                    score_cfg_for(condition_cfg), retention_cfg_for(cond, self.cfg, seed), model.case_count(),
                    retrieval_epochs=re, adapter_epochs=ae, lr_scale=self.cfg["continuation"]["lr_scale"], seed=seed)
                rev = evaluate_adapted(random_model, random_ad)
                random_loss = reference_loss(random_model, maintenance_reference(data, condition_cfg, random_ad))
                random_info["final_reference_loss"] = random_loss
                random_info["cumulative_loss_increase"] = random_loss - result["summary"]["original_reference_loss"]
                random_info["within_adaptive_loss_budget"] = random_info["cumulative_loss_increase"] <= retention_cfg_for(cond, self.cfg, seed).allowed_loss_increase + 1e-10
                final_store = CaseStatisticsStore(data.task_type)
                audit(model, data, condition_cfg, final_store, len(result["events"]), adapter=ad)
                metrics = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)},
                    **{f"matched_full_test_{k}": v for k, v in cev.items() if not isinstance(v, dict)},
                    **{f"matched_random_test_{k}": v for k, v in rev.items() if not isinstance(v, dict)},
                    "n_cases": model.case_count(), "matched_full_n_cases": control.case_count()}
                self.finish(ds=ds, seed=seed, cond_label=label, cond=cond, data=data,
                    model=model, store=final_store, archive=archive, metrics=metrics,
                    history={"core": tr.history, "adapter": info["history"], "matched_full": control_history,
                             "matched_random": random_info["history"]},
                    events=result["events"], components={"retrieve": "NN-kNN core", "reuse_or_adapter": actual_mode,
                        "revise": "off", "retain": "adapted retrained full removal", "mcb": "off",
                        "component_synchronization": cond["scope"]},
                    budgets={"case_capacity": model.case_count(), "accepted_optimizer_updates": result["summary"]["optimizer_updates"]["accepted"],
                        "retrieval_epochs_per_trial": re, "adapter_epochs_per_trial": ae,
                        "matching": "equal accepted phase blocks and examples; capacity and cost differ"},
                    maintenance={"policy": "removal_influence", "reference_stream": reference_stream,
                        "selection_loss": "final_prediction_only", "adapter_training_loss": rc.loss},
                    model_desc=model_desc, prediction_adapter=ad,
                    optimizer=opt, adapter_optimizer_state=ast,
                    optimizer_checkpoint_epoch={"retrieval": "fixed_final", "adapter": "fixed_final"},
                    extra_files={"maintenance_summary.json": result["summary"], "maintenance_scoring_trace.json": result["scoring_trace"],
                                 "matched_random_summary.json": random_info},
                    binary_files={"initial.pt": checkpoint(base, initial_adapter, tr.optimizer, info["optimizer_state"], initial=True),
                        "matched_full.pt": checkpoint(control, cad, copt, cast),
                        "matched_random.pt": checkpoint(random_model, random_ad, random_opt, random_ast)})

    def run_reuse_retention(self, ds, seed):
        """Matched maintenance with a trained frozen adapter and final-outcome evidence."""
        data = self.data_for(ds, seed)
        if data.task_type != "classification" or self.cfg["protocol"]["finetune_epochs"] != 0:
            raise ValueError("reuse_retention requires classification and frozen parameters (no finetune)")
        rc = ReuseConfig(seed=seed, **self.cfg["reuse"])

        def adapter_stage(core):
            adapter, info = train_classification_adapter(core["model"], data, rc)
            return {"adapter": adapter, "info": info}
        units = [(f"{cond['policy']}_K{kspec}", {**cond, "K": kspec}) for cond in self.cfg["conditions"] for kspec in self.cfg["K"]]
        core = self.core_stage(ds, seed, data, units, extra=adapter_stage)
        if core is not None:
            base, cc, tr, adapter, info = core["model"], core["cc"], core["tr"], core["adapter"], core["info"]
        for cond in self.cfg["conditions"]:
            for kspec in self.cfg["K"]:
                if self.skip(ds, seed, f"{cond['policy']}_K{kspec}", {**cond, "K": kspec}):
                    continue
                model = copy.deepcopy(base)
                store, archive = CaseStatisticsStore(), CaseArchive()
                ainfo = audit(model, data, self.cfg, store, 0, adapter=adapter)
                label = f"{cond['policy']}_K{kspec}"
                res = run_maintenance(model, store, archive, resolve_K(kspec, model.case_count()),
                                      retention_cfg_for(cond, self.cfg, seed), score_cfg_for(self.cfg),
                                      step=0, run_id=label, reg_bins=data.reg_bins,
                                      reference=maintenance_reference(data, self.cfg, adapter))
                ev = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc, query_nominal=encoded_queries(adapter, data, "test"), test_groups=data.test_groups)
                metrics = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)},
                           **{f"test_flip_{k}": v for k, v in ev["flips"].items()},
                           "n_cases": model.case_count(), "audit_mean_loss_final": ainfo["mean_loss_final"]}
                self.finish(ds=ds, seed=seed, cond_label=label, cond={**cond, "K": kspec},
                            data=data, model=model, store=store, archive=archive, metrics=metrics,
                            history={"core": tr.history, "adapter": info["history"]}, events=res["events"],
                            components={"retrieve": "frozen NN-kNN core", "reuse_or_adapter": rc.output_mode,
                                        "revise": "off", "retain": cond["policy"], "mcb": "off", "component_synchronization": "off"},
                            budgets={"case_capacity": resolve_K(kspec, len(data.y_train)), "finetune_epochs": 0},
                            maintenance={"policy": cond["policy"], "reference_stream": self.cfg["statistics"]["source"],
                                         "thresholds_and_smoothing": self.cfg["statistics"]},
                            model_desc={"core": cc.__dict__, "adapter": rc.__dict__}, prediction_adapter=adapter,
                            adapter_optimizer_state=info["optimizer_state"],
                            optimizer_checkpoint_epoch={"retrieval": None, "adapter": info["best_epoch"]},
                            extra_files={"maintenance_scoring_trace.json": res.get("scoring_trace", []),
                                         "maintenance_summary.json": res["summary"]})

    def run_removal_retraining(self, ds, seed):
        """Full-query retrained removal plus frozen and matched-training controls."""
        from model.t1.retraining import continued_trial, reference_loss, run_retrained_removal

        data = self.data_for(ds, seed)
        units = [(f"{t}_K{kspec}", {"treatment": t, "K": kspec}) for kspec in self.cfg["K"] for t in self.cfg["treatments"]]
        core = self.core_stage(ds, seed, data, units)
        if core is not None:
            base, cc, tr = core["model"], core["cc"], core["tr"]
            starting = {"model_state": base.state_dict(), "optimizer_state": tr.optimizer.state_dict(), "core": cc.__dict__}
        # every treatment unit keeps the starting weights as its named state starting_checkpoint
        initial = "weights/starting_checkpoint.pt"
        budget = self.cfg["retraining"]
        for kspec in self.cfg["K"]:
            if core is not None:
                K = resolve_K(kspec, base.case_count())
                ref = maintenance_reference(data, self.cfg)
                scfg = score_cfg_for(self.cfg)
                rcfg = retention_cfg_for({"policy": "removal_influence"}, self.cfg, seed)

                def retrained():  # the retrained removal of this K: shared by its treatments, a stage of its own
                    store = CaseStatisticsStore(data.task_type)
                    audit(base, data, self.cfg, store, 0)
                    model, _, archive, res = run_retrained_removal(
                        base, tr.optimizer, data, cc, ref, store, scfg, rcfg, K,
                        epochs=int(budget["epochs_per_candidate"]), lr_scale=float(budget["lr_scale"]),
                        run_id=f"retrained-K{kspec}-s{seed}")
                    return {"model": model, "archive": archive, "res": res}
                removal = self.stage(ds, seed, f"retrained_K{kspec}", [u for u in units if u[1]["K"] == kspec], retrained)
                if removal is not None:
                    matched_epochs = removal["res"]["summary"]["accepted_model_training_epochs"]
            for treatment in self.cfg["treatments"]:
                if self.skip(ds, seed, f"{treatment}_K{kspec}", {"treatment": treatment, "K": kspec}):
                    continue
                model = copy.deepcopy(base)
                opt = clone_optimizer(tr.optimizer, model, cc)
                local_store, archive = CaseStatisticsStore(data.task_type), CaseArchive()
                audit(model, data, self.cfg, local_store, 0)
                history = []
                label = f"{treatment}_K{kspec}"
                extra_seconds = 0.0
                if treatment == "removal_retrained":
                    model, archive, res = removal["model"], removal["archive"], removal["res"]
                else:
                    policies = {"full_frozen": "full_memory", "full_matched": "full_memory",
                                "random_frozen": "random", "random_matched": "random",
                                "removal_frozen": "removal_influence"}
                    if treatment not in policies:
                        raise ValueError(f"Unknown retraining treatment: {treatment}")
                    t0 = time.perf_counter()
                    res = run_maintenance(model, local_store, archive, K,
                                          retention_cfg_for({"policy": policies[treatment]}, self.cfg, seed),
                                          scfg, step=0, run_id=label, optimizer=opt,
                                          reg_bins=data.reg_bins, reference=ref)
                    res["summary"]["selection_seconds"] = time.perf_counter() - t0
                    if treatment.endswith("matched") and matched_epochs:
                        t0 = time.perf_counter()
                        model, _, ft = continued_trial(model, opt, data, cc, epochs=matched_epochs,
                                                       lr_scale=float(budget["lr_scale"]))
                        history = ft.history
                        extra_seconds = time.perf_counter() - t0
                # Refresh final statistics: candidate selection never uses test labels.
                ainfo = audit(model, data, self.cfg, local_store, 1)
                extra_epochs = matched_epochs if treatment.endswith("matched") or treatment == "removal_retrained" else 0
                m = {**test_metrics(model, data), "n_cases": model.case_count(), "requested_K": K,
                     "extra_training_epochs": extra_epochs, "matched_training_epochs": matched_epochs,
                     "reference_loss_final": reference_loss(model, ref),
                     "reference_loss_initial": reference_loss(base, ref),
                     "audit_mean_loss_final": ainfo["mean_loss_final"],
                     "selection_seconds": res["summary"]["selection_seconds"],
                     "extra_training_seconds": extra_seconds,
                     "capacity_matched": model.case_count() == K,
                     **truth_retention(model, data)}
                self.finish(ds=ds, seed=seed, cond_label=label, cond={"treatment": treatment, "K": kspec},
                            data=data, model=model, store=local_store, archive=archive, metrics=m,
                            history={"core": tr.history, "extra_training": history}, events=res["events"],
                            components={"retrieve": "NN-kNN retrieval-only core", "reuse_or_adapter": "off",
                                        "revise": "off", "retain": treatment, "mcb": "on" if cc.mcb_enabled else "off",
                                        "component_synchronization": "off"},
                            budgets={"case_capacity": K, "epochs_per_candidate": budget["epochs_per_candidate"],
                                     "accepted_extra_epochs": extra_epochs, "matched_control_epochs": matched_epochs,
                                     "candidate_training_epochs_total": res["summary"].get("candidate_training_epochs_total", 0),
                                     "core_epochs_run": tr.epochs_run},
                            maintenance={"policy": treatment, "reference_stream": ref.stream,
                                         "starting_checkpoint": str(initial), "parameters": budget,
                                         "allowed_loss_increase": rcfg.allowed_loss_increase},
                            model_desc={"core": cc.__dict__},
                            extra_files={"maintenance_summary.json": res["summary"],
                                         "maintenance_scoring_trace.json": res.get("scoring_trace", [])},
                            binary_files={"starting_checkpoint.pt": starting})

    def run_sync(self, ds, seed):
        for cond in self.cfg["conditions"]:
            sync_maintenance_for(cond, self.cfg, seed)  # refuse a bad condition before any training
        data = self.data_for(ds, seed)
        for arm in self.arms():
            units = [self.sync_unit(cond, arm) for cond in self.cfg["conditions"]]
            core = self.core_stage(ds, seed, data, units, arm)
            near_scale = None if core is None else near_scale_of(core["model"])
            for cond in self.cfg["conditions"]:
                if not self.skip(ds, seed, *self.sync_unit(cond, arm)):
                    self.sync_condition(ds, seed, data, core["model"], core["cc"], core["tr"], cond, arm, near_scale)

    def sync_condition(self, ds, seed, data, base, cc, tr, cond, arm=None, near_scale=None):
        """One synchronized condition continuing ``base`` or, with ``core_pretrain: false``,
        training retrieval and reuse together from the untrained initialization."""
        fields = SyncConfig.__dataclass_fields__
        rcfg, k_spec, m_epochs = sync_maintenance_for(cond, self.cfg, seed)
        merged = {**self.cfg["sync"], **cond}
        pretrained = bool(merged.get("core_pretrain", True))
        if pretrained:
            model = copy.deepcopy(base)
            t_star = tr.best_epoch
        else:
            model = build_model(data.X_train, data.y_train, cc, data.num_classes)
            near_scale = near_scale_of(model)
            t_star = 0
        radius_method = merged.get("free_radius_method", "trained_bias_region")
        initial_bias = None
        if radius_method == "initial_bias_region":
            initial_bias = build_model(data.X_train, data.y_train, cc, data.num_classes).biases.detach().clone()
        calibration_snapshots = []
        def calibrate(ep):
            ids = model.active_case_ids().long()
            radius, snapshot = calibrate_free_radius(model, s_task=float(merged["s_task"]),
                snapshot_step=ep, method=radius_method, neighbor_k=cc.bias_init_k,
                aggregation=merged.get("radius_aggregation", "mean"),
                bias_transform=merged.get("bias_transform", "identity"),
                shrink_factor=float(merged.get("shrink_factor", .5)),
                clipping_quantile=float(merged.get("clipping_quantile", .95)),
                initial_bias=None if initial_bias is None else initial_bias[ids])
            snapshot["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            snapshot["active_case_count"] = model.case_count()
            calibration_snapshots.append(compact_calibration_snapshot(snapshot))
            return radius
        fr = calibrate(t_star)
        merged.setdefault("output_mode", "aggregate_regression" if data.task_type == "regression" else "nominal_residual_scores")
        sc = SyncConfig(seed=seed, **{k: (tuple(v) if k == "hidden_dims" else v) for k, v in merged.items() if k in fields and k != "seed"})
        store, archive, events = CaseStatisticsStore(data.task_type), CaseArchive(), []
        capacity_match = self.cfg.get("capacity_match")
        capacity_spec = capacity_match["K"] if capacity_match else self.cfg["sync"].get("K", 1.0)
        K = resolve_K(k_spec, model.case_count()) if "rrr" in sc.schedule else None

        def hook(ep, mdl, opt, adapter):
            audit(mdl, data, self.cfg, store, ep, adapter=adapter)
            res = run_maintenance(mdl, store, archive, K, rcfg, score_cfg_for(self.cfg), step=ep, run_id=cond["schedule"], optimizer=opt, reg_bins=data.reg_bins, reference=maintenance_reference(data, self.cfg, adapter))
            events.extend(res["events"])
            return res["summary"]

        recal = calibrate
        # A trained core continues its Adam state (see clone_optimizer); an
        # untrained one starts a fresh optimizer.
        ropt = clone_optimizer(tr.optimizer, model, cc) if pretrained else make_optimizer(model, cc)
        for grp in ropt.param_groups:
            grp["lr"] = grp["lr"] * float(merged["retrieval_lr_scale"])
        tracker = StabilityTracker(data.X_val, k=cc.top_k)
        tracker.snapshot(model, 0, model.active_case_ids().tolist())
        def observe(ep, mdl, adapter):
            return tracker.snapshot(mdl, ep, mdl.active_case_ids().tolist())
        adapter, info = train_synchronized(model, data, cc, sc, fr, near_scale=near_scale, maintenance_hook=hook, maintenance_epochs=m_epochs, recalibrate=recal, core_optimizer=ropt, epoch_observer=observe)
        model_desc = {"core": cc.__dict__, "sync": sc.__dict__, "near_scale": near_scale,
            "core_pretrain": pretrained, "retrieval_lr_scale": float(merged["retrieval_lr_scale"]),
            "post_selection_capacity_match": capacity_match,
            "free_radius_at_t_star": fr.to_dict(), "free_radius_selected": info["free_radius"]}
        binary_files = {"calibration_snapshots.pt": calibration_snapshots}
        extra_files = {"calibration_protocol.json": {
            "method": radius_method, "pair_source": "active_training_cases_only_nonself",
            "initial_bias_region_metric": "post_core_learned_metric" if initial_bias is not None else None,
            "snapshot_file": "calibration_snapshots.pt",
            "snapshot_steps": [s["free_radius"]["snapshot_step"] for s in calibration_snapshots],
            "recompute": "declared_maintenance_checkpoints_only"}}
        if capacity_match:
            binary_files["before_capacity_match.pt"] = {
                "model_state": copy.deepcopy(model.state_dict()), "active_case_count": model.case_count(),
                "adapter_state": copy.deepcopy(adapter.state_dict()), "optimizer_state": copy.deepcopy(ropt.state_dict()),
                "adapter_optimizer_state": copy.deepcopy(info["adapter_optimizer_state"]),
                "model_description": copy.deepcopy(model_desc),
                "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
                "adapter_requires_grad": {n: p.requires_grad for n, p in adapter.named_parameters()},
                "optimizer_checkpoint_epoch": {"retrieval": info["best_epoch"], "adapter": info["best_epoch"]}}
            audit(model, data, self.cfg, store, info["best_epoch"], adapter=adapter)
            matched_K = resolve_K(capacity_match["K"], base.case_count())
            match_res = run_maintenance(model, store, archive, matched_K,
                retention_cfg_for({"policy": capacity_match["policy"]}, self.cfg, seed), score_cfg_for(self.cfg),
                step=info["best_epoch"] + 1, run_id=f"{sc.schedule}-posthoc",
                optimizer=ropt, reg_bins=data.reg_bins, reference=maintenance_reference(data, self.cfg, adapter))
            events.extend(match_res["events"])
            extra_files["capacity_match_summary.json"] = match_res["summary"]
            extra_files["capacity_match_scoring_trace.json"] = match_res.get("scoring_trace", [])
            extra_files["capacity_match_protocol.json"] = {
                "target_capacity": matched_K, "policy": capacity_match["policy"],
                "after_selected_checkpoint": True, "finetune_epochs": 0,
                "matching": "final capacity; executed phases and retrieval cost separately reported",
                "achieved": model.case_count() == matched_K}
        rc = None
        if data.task_type == "classification":
            rc = ReuseConfig(output_mode=sc.output_mode, loss="combined", lambda_diff=1.0, lambda_cls=1.0, probability_mode=self.cfg["sync"].get("probability_mode", "softmax"), seed=seed,
                             score_scale=sc.score_scale)
        ev = adapted_metrics(model, adapter, data, "test", self.cfg, rc)
        vev = adapted_metrics(model, adapter, data, "val", self.cfg, rc)
        audit(model, data, self.cfg, store, info["best_epoch"], adapter=adapter)
        m = {**{f"test_{k}": v for k, v in ev.items() if not isinstance(v, dict)},
             **{f"test_flip_{k}": v for k, v in ev.get("flips", {}).items()},
             "tau_task": fr.tau_task, "free_radius_status": fr.status,
             "tau_task_final": info["free_radius"]["tau_task"], "n_cases": model.case_count(),
             **{f"val_{k}": v for k, v in vev.items() if not isinstance(v, dict)},
             "best_val_L_post": info["best_val_L_post"], "sync_best_epoch": info["best_epoch"],
             **online_ema_gap(model),
             "best_val_err": info["best_val_err"], "sync_val_L_pre_epoch0": info["v_pre0"],
             "sync_val_err_pre_epoch0": info["v_err_pre0"], "sync_guard_rejected_epochs": info["guard_rejected_epochs"],
             "sync_no_eligible_checkpoint": info["no_eligible_checkpoint"],
             "test_locality_input": input_locality(model, data.X_test)}
        for key in ("representation_drift", "neighborhood_churn", "selection_churn"):
            values = [r[key] for r in tracker.records if key in r]
            if values:
                m[f"mean_{key}"] = float(np.mean(values))
        phase_updates = {phase: sum(h["optimizer_updates"] for h in info["history"] if h["phase"] == phase)
                         for phase in ("retrieval", "adapter", "joint")}
        selected_updates = {phase: sum(h["optimizer_updates"] for h in info["history"]
            if h["phase"] == phase and h["epoch"] <= info["best_epoch"]) for phase in ("retrieval", "adapter", "joint")}
        label = self.sync_unit(cond, arm)[0]
        maintenance = {"policy": (self.cfg["retention"]["policy"] if rcfg is None else rcfg.policy) if events else "none"}
        if rcfg is not None:  # this condition's effective in-training settings (rows carry only its own keys)
            maintenance["in_training"] = {"K": K, "K_spec": k_spec, "maintenance_epochs": sorted(m_epochs),
                                          "retention": dict(rcfg.__dict__)}
        self.finish(ds=ds, seed=seed, cond_label=label, cond={**arm_fields(arm), **cond}, data=data, model=model, store=store, archive=archive, metrics=m, history={"core": tr.history if pretrained else [], "sync": info["history"]}, events=events,
                    components={"retrieve": "NN-kNN core", "reuse_or_adapter": sc.output_mode, "revise": "off", "retain": "checkpoint maintenance" if events else "none", "mcb": "on" if cc.mcb_enabled else "off", "component_synchronization": sc.schedule},
                    budgets={"case_capacity": model.case_count(), "sync_epochs_max": sc.epochs,
                        "core_pretrain": pretrained, "core_epochs_run": tr.epochs_run if pretrained else 0,
                        "sync_epochs_run": len(info["history"]), "executed_optimizer_updates": phase_updates,
                        "selected_optimizer_updates": selected_updates,
                        "training_retrieval_query_case_pairs": sum(h["retrieval_query_case_pairs"] for h in info["history"]),
                        "sync_elapsed_seconds_including_maintenance_and_validation": info["elapsed_seconds"],
                        "posthoc_finetune_epochs": 0}, maintenance=maintenance,
                    model_desc=model_desc,
                    prediction_adapter=adapter, optimizer=ropt,
                    binary_files=binary_files, extra_files=extra_files,
                    adapter_optimizer_state=info["adapter_optimizer_state"],
                    optimizer_checkpoint_epoch={"retrieval": info["best_epoch"], "adapter": info["best_epoch"]})

    def run_mcb(self, ds, seed):
        data = self.data_for(ds, seed)
        for cond in self.cfg["conditions"]:
            if self.skip(ds, seed, f"mcb{int(bool(cond['mcb']))}_{cond['policy']}_K{cond['K']}", cond):
                continue
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
                res = run_maintenance(mdl, store, archive, K, rcfg, scfg, step=ep, run_id=f"mcb{cond['mcb']}", optimizer=opt, reg_bins=data.reg_bins, reference=maintenance_reference(data, self.cfg))
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
        true_labels = data.meta.get("true_labels")
        if true_labels is None:
            raise ValueError("synthetic data must expose true labels for the simulated reviewer")
        rv = self.cfg["revise"]
        units = [(f"{method}_b{b}", {"method": method, "review_budget": b}) for method in rv["methods"] for b in rv["budgets"]]
        core = self.core_stage(ds, seed, data, units)

        def ablation():  # every method and budget in one call: the revise stage
            model, cc, tr = core["model"], core["cc"], core["tr"]
            store = CaseStatisticsStore()
            audit(model, data, self.cfg, store, 0)
            ids = model.active_case_ids().cpu().numpy()
            scores = score_cases(ids, active_cohorts(model), model.biases[: model.case_count()].detach().cpu().numpy(), store, score_cfg_for(self.cfg))
            truth = data.case_truth["corrupted"]
            return {"rows": flagging_ablation(model, data, scores, cc, truth_corrupted=truth, true_labels=np.asarray(true_labels), budgets=rv["budgets"], methods=rv["methods"], retrain_epochs=rv["retrain_epochs"], run_id=f"{self.exp}-s{seed}", seed=seed, core_optimizer=tr.optimizer, retrain_lr_scale=float(rv["retrain_lr_scale"]), capture_artifacts=True, matched_training_control=bool(rv.get("matched_training_control", False)))}
        revised = None if core is None else self.stage(ds, seed, "revise", units, ablation)
        if revised is None:  # every run is complete: restore their RNG states in order
            for label, cond in units:
                self.skip(ds, seed, label, cond)
            return
        cc, tr = core["cc"], core["tr"]
        for r in revised["rows"]:
            if self.skip(ds, seed, f"{r['method']}_b{r['review_budget']}", {"method": r["method"], "review_budget": r["review_budget"]}):
                continue
            artifacts = r.pop("_artifacts")
            final_model = artifacts["final_model"]
            revised_data = copy.copy(data)
            revised_data.y_train = artifacts["y_train_corrected"]
            final_store = CaseStatisticsStore()
            audit(final_model, revised_data, self.cfg, final_store, rv["retrain_epochs"])
            stages = artifacts["stages"]
            for stage, checkpoint in stages.items():
                checkpoint["model_description"] = {"core": cc.__dict__, "stage": stage}
                checkpoint["evaluation_metrics"] = r[stage]
                checkpoint["training_targets"] = data.y_train if stage in {"M0", "MC"} else artifacts["y_train_corrected"]
            label = f"{r['method']}_b{r['review_budget']}"
            m = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for ck in stages:
                for k, v in r[ck].items():
                    m[f"test_{ck}_{k}"] = v
            for fk in (key for key in r if "flips_" in key):
                if fk in r:
                    for k, v in r[fk].items():
                        m[f"{fk}_{k}"] = v
            self.finish(ds=ds, seed=seed, cond_label=label, cond={"method": r["method"], "review_budget": r["review_budget"]}, data=data, model=final_model, store=final_store, archive=None, metrics=m, history={"core": tr.history, "retrain": artifacts["history"], "matched_control": artifacts.get("control_history", [])}, events=[],
                        components={"retrieve": "NN-kNN core", "reuse_or_adapter": "off", "revise": "simulated oracle review (T1.2 harness)", "retain": "none", "mcb": "off", "component_synchronization": "off"},
                        budgets={"review_budget": r["review_budget"], "retrain_epochs": rv["retrain_epochs"], "matched_control_epochs": r.get("matched_control_epochs", 0)}, maintenance={"policy": "none"}, model_desc={"core": cc.__dict__, "stage": "M2", "statistics_training_targets": "M2.pt:training_targets"}, extra_files={"interventions.json": r["interventions"]}, binary_files={**{f"{stage}.pt": checkpoint for stage, checkpoint in stages.items()}, "intervention_state.pt": artifacts["intervention_log"]}, optimizer=artifacts["final_optimizer"])


def check_unit_lengths(ap, runner, cfg, seeds, only_dataset) -> None:
    """Refuse a study whose unit paths are too long even for a one-character condition label; warn when a label
    of the longest kept length (``records.MAX_COMPONENT``, longer ones are hashed down to it) would be too long,
    since the condition labels are only known when each condition is reached, after its training."""
    from model.common.tracking import MAX_PATH
    dsets = [ds for ds in cfg["datasets"] if not only_dataset or ds["name"] == only_dataset]
    seed = f"s{max(seeds, key=lambda s: len(str(s)))}" if seeds else "s0"
    try:
        records.check_unit_paths(runner.study, [(runner.exp, runner.dataset_key(ds), "x", seed) for ds in dsets])
    except ValueError as exc:
        ap.error(f"--study: {exc}")
    longest = max((records.event_path_length(runner.study, runner.exp, runner.dataset_key(ds), "x" * records.MAX_COMPONENT, seed)
                   for ds in dsets), default=0)
    if longest >= MAX_PATH:
        print(f"[warning] a condition label longer than {records.MAX_COMPONENT - (longest - MAX_PATH + 1)} characters "
              f"would give an event file path over the limit ({MAX_PATH}) after its training; a shorter study name avoids it",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=str(REPO / "results" / "t1_pi20260920"),
                    help="inputs only: the source snapshot zip and <dataset>/data_snapshot_s<seed>.pt")
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--study", required=True,
                    help="every run is recorded as a unit under NN-KNN/runs/t1/<study>/<experiment>/ (or $NNKNN_RUNS)")
    args = ap.parse_args()
    try:  # the run root and the study name, before any data is loaded
        records.check_study(args.study)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(f"--study: {exc}")
    if os.environ.get("NNKNN_DETERMINISTIC") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS
        torch.use_deterministic_algorithms(True)
    cfg = load_config(Path(args.config))
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    runner = Runner(cfg, Path(args.out), args.study)
    fn = getattr(runner, f"run_{cfg['type']}")
    seeds = args.seeds if args.seeds else cfg["seeds"]
    check_unit_lengths(ap, runner, cfg, seeds, args.only_dataset)
    for ds in cfg["datasets"]:
        if args.only_dataset and ds["name"] != args.only_dataset:
            continue
        for seed in seeds:
            t0 = time.time()
            runner.check_recorded(ds, int(seed))
            fn(ds, int(seed))
            print(f"[timing] {cfg['experiment']} {ds['name']} s{seed}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
