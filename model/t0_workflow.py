from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.nn_cdh import (
    ClassificationNNCDHAdapter,
    compute_classification_adaptation_inputs,
)
from model.nnknn_model import NN_KNN_Model, default_args
from model.t0_maintenance import (
    BiasOnlyPolicy,
    CaseArchiveStore,
    CaseMaintenancePolicy,
    CaseStatistics,
    CaseStatisticsStore,
    CoreSetGreedyPolicy,
    DROP3Policy,
    ICFPolicy,
    MaintenanceAction,
    ProvenanceBiasCoveragePolicy,
    ProvenanceOnlyPolicy,
    StratifiedRandomPolicy,
    TrustworthinessOnlyPolicy,
    audit_regression_provenance_counterfactual,
    write_maintenance_artifacts,
)



@dataclass
class T0Config:
    """Configuration for T0 full-cycle neural CBR experiments."""
    task_type: str = "classification"
    case_capacity: int = 200
    target_capacity: int = 100
    case_maintenance_policy: str = "provenance_bias_coverage"  # or 'bias_only', 'provenance_only', 'trustworthiness_only', 'stratified', 'full_memory'
    case_maintenance_frequency: int = 25  # Maintenance check every N epochs
    case_score_smoothing: float = 1.0
    case_trust_alpha: float = 0.5
    case_min_retrieval_count: int = 2
    case_min_activation_mass: float = 0.05
    case_min_per_cohort: int = 2
    case_archive_evictions: bool = True
    case_revision_enabled: bool = False

    # Classification reuse extension
    classification_adapter_enabled: bool = False
    classification_adapter_output_mode: str = "nominal_residual_scores"  # or 'logit_residual'
    classification_adapter_hidden_dims: tuple[int, int] = (64, 32)
    classification_adapter_freeze_retrieval_first: bool = True
    classification_adapter_joint_finetune: bool = False
    joint_retrieval_lr: float = 1e-4
    classification_adapter_lambda_diff: float = 1.0
    classification_adapter_lambda_cls: float = 1.0
    classification_adapter_lambda_mag: float = 0.01

    # Regression counterfactual options
    regression_counterfactual_method: str = "counterfactual"  # or 'directional'

    # Training parameters
    learning_rate: float = 1e-3
    adapter_learning_rate: float = 5e-4
    epochs: int = 100
    adapter_epochs: int = 40
    batch_size: int = 128
    top_k: int = 10
    tau: float = 1.0
    case_normalizer: str = "softmax"
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_t0_maintenance_policy(cfg: T0Config) -> CaseMaintenancePolicy:
    """Factory for T0 case maintenance policies."""
    mode = cfg.case_maintenance_policy.strip().lower()
    if mode in {"provenance_bias_coverage", "t0_primary", "default"}:
        return ProvenanceBiasCoveragePolicy(
            smoothing=cfg.case_score_smoothing,
            alpha=cfg.case_trust_alpha,
            min_retrieval_count=cfg.case_min_retrieval_count,
            min_activation_mass=cfg.case_min_activation_mass,
            min_per_cohort=cfg.case_min_per_cohort,
        )
    elif mode in {"bias_only", "current_pruning"}:
        return BiasOnlyPolicy()
    elif mode in {"provenance_only", "q_only"}:
        return ProvenanceOnlyPolicy(smoothing=cfg.case_score_smoothing)
    elif mode in {"trustworthiness_only", "t_only"}:
        return TrustworthinessOnlyPolicy(
            smoothing=cfg.case_score_smoothing,
            alpha=cfg.case_trust_alpha,
        )
    elif mode in {"stratified", "random"}:
        return StratifiedRandomPolicy(seed=cfg.seed)
    elif mode in {"drop3", "drop_3"}:
        return DROP3Policy(k_neighbors=3)
    elif mode in {"icf"}:
        return ICFPolicy(k_neighbors=3)
    elif mode in {"coreset", "coreset_greedy", "kcenter"}:
        return CoreSetGreedyPolicy(seed=cfg.seed)
    elif mode in {"full_memory", "none"}:
        return BiasOnlyPolicy()  # target_capacity will equal full memory
    else:
        raise ValueError(f"Unknown case_maintenance_policy: {cfg.case_maintenance_policy}")


def build_t0_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: T0Config,
    feature_extractor: nn.Module | None = None,
    nominal_dim: int = 0,
) -> tuple[NN_KNN_Model, CaseStatisticsStore, CaseArchiveStore, CaseMaintenancePolicy]:
    """Construct an NN_KNN_Model equipped with T0 stable IDs, statistics, archive, and policy."""
    num_cases = X_train.size(0)
    is_regression = (cfg.task_type.lower() == "regression")
    
    if is_regression:
        num_classes = 1
        if y_train.dim() == 1:
            y_model_labels = y_train.view(-1, 1).float()
        else:
            y_model_labels = y_train.float()
        y_np = y_model_labels.view(-1).detach().cpu().numpy()
        # Quantile binning for cohort comparable bias and coverage
        n_bins = min(5, len(np.unique(y_np)))
        if n_bins > 1:
            quantiles = np.quantile(y_np, np.linspace(0, 1, n_bins + 1))
            cohort_assignments = np.digitize(y_np, quantiles[1:-1]).tolist()
        else:
            cohort_assignments = [0] * num_cases
    else:
        num_classes = y_train.size(1) if y_train.dim() > 1 else int(y_train.max().item() + 1)
        if y_train.dim() == 1:
            y_model_labels = F.one_hot(y_train.long(), num_classes=num_classes).float()
        else:
            y_model_labels = y_train.float()
        cohort_assignments = [int(y_model_labels[i].argmax().item()) for i in range(num_cases)]

    model_config = copy.deepcopy(default_args)
    model_config.update({
        "task_type": cfg.task_type,
        "tau": cfg.tau,
        "top_k": min(cfg.top_k, num_cases),
        "case_normalizer": cfg.case_normalizer,
        "active_case_count": num_cases,
        "explanation_mode": True,
    })

    model = NN_KNN_Model(
        cases=X_train.clone(),
        labels=y_model_labels.clone(),
        feature_extractor=feature_extractor,
        **model_config,
    )

    # Initialize stable case_ids
    device = X_train.device
    model.register_buffer("case_ids", torch.arange(num_cases, dtype=torch.long, device=device))
    model.register_buffer("next_case_id", torch.tensor(num_cases, dtype=torch.long, device=device))

    # Initialize T0 stores and policy
    stats_store = CaseStatisticsStore()
    archive_store = CaseArchiveStore()
    policy = make_t0_maintenance_policy(cfg)

    # Register initial case statistics
    for i in range(num_cases):
        cohort = cohort_assignments[i]
        stats_store.register_case(
            case_id=i,
            initial_bias=float(model.biases[i].item()),
            cohort_id=cohort,
            insertion_step=0,
            protected=(i < cfg.case_min_per_cohort),
        )

    # Attach classification adapter if enabled
    if cfg.classification_adapter_enabled:
        feature_dim = (
            int(feature_extractor.feature_dim)
            if feature_extractor is not None and hasattr(feature_extractor, "feature_dim")
            else X_train.size(1)
        )
        adapter = ClassificationNNCDHAdapter(
            feature_dim=feature_dim,
            num_classes=num_classes,
            nominal_dim=nominal_dim,
            hidden_dims=cfg.classification_adapter_hidden_dims,
            output_mode=cfg.classification_adapter_output_mode,
        )
        model.classification_adapter = adapter

    return model, stats_store, archive_store, policy


def train_leave_one_out_adapter(
    model: NN_KNN_Model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: T0Config,
    nominal_train: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Train classification NN-CDH adapter on leave-one-out retrieval neighborhoods."""
    if getattr(model, "classification_adapter", None) is None:
        return {"status": "skipped_no_adapter"}

    adapter = model.classification_adapter.to(device)
    model.to(device)
    model.eval()

    num_samples = X_train.size(0)
    num_classes = adapter.num_classes
    y_idx = y_train.long().to(device) if y_train.dim() == 1 else y_train.argmax(dim=-1).to(device)
    y_one_hot = F.one_hot(y_idx, num_classes=num_classes).float()

    joint_finetune = bool(getattr(cfg, "classification_adapter_joint_finetune", False))
    if joint_finetune:
        model.train()
        optimizer = torch.optim.Adam([
            {"params": adapter.parameters(), "lr": cfg.adapter_learning_rate},
            {"params": [model.biases], "lr": getattr(cfg, "joint_retrieval_lr", 1e-4)},
        ])
    else:
        model.eval()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.adapter_learning_rate)
    batch_size = min(cfg.batch_size, num_samples)

    loss_history = []
    adapter.train()

    for epoch in range(cfg.adapter_epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = X_train[batch_idx].to(device)
            yb = y_one_hot[batch_idx].to(device)
            yb_idx = y_idx[batch_idx]

            # Forward query features
            with torch.no_grad():
                q_feats = model.feature_extractor(xb) if model.feature_extractor is not None else xb
                # Leave-one-out distance computation
                active_count = model.case_count()
                c_feats = (
                    model.feature_extractor(model.cases[:active_count])
                    if model.feature_extractor is not None
                    else model.cases[:active_count]
                )
                q_exp = q_feats.unsqueeze(1).expand(-1, active_count, -1)
                c_exp = c_feats.unsqueeze(0).expand(xb.size(0), -1, -1)
                dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))
                scores = model.biases[:active_count].unsqueeze(0) - dists

                # Mask self-retrieval: query i cannot retrieve case i
                for k_pos, orig_idx in enumerate(batch_idx):
                    if orig_idx < active_count:
                        scores[k_pos, orig_idx] = float("-inf")

                # Top-K
                k_eff = min(cfg.top_k, active_count - 1)
                top_scores, top_idx = torch.topk(scores, k=k_eff, dim=1)
                w = F.softmax(top_scores / cfg.tau, dim=1)

                # Aggregate retrieved quantities
                top_case_feats = c_feats[top_idx]  # [B, K, D]
                top_labels = model.labels[top_idx]  # [B, K, C]

                nom_q = nominal_train[batch_idx].to(device) if nominal_train is not None else None
                nom_ret = None
                if nominal_train is not None:
                    nom_ret = nominal_train[:active_count][top_idx].to(device)

                Delta_z_q, p0_q, Delta_u_q = compute_classification_adaptation_inputs(
                    query_features=q_feats,
                    retrieved_features=top_case_feats,
                    case_weights=w,
                    retrieved_labels=top_labels,
                    nominal_query=nom_q,
                    nominal_retrieved=nom_ret,
                )

            # Optimization step on adapter
            optimizer.zero_grad()
            r_hat, s_q = adapter(Delta_z_q, p0_q, Delta_u_q)
            loss_dict = adapter.compute_loss(
                r_hat,
                s_q,
                p0_q,
                yb_idx,
                lambda_diff=cfg.classification_adapter_lambda_diff,
                lambda_cls=cfg.classification_adapter_lambda_cls,
                lambda_mag=cfg.classification_adapter_lambda_mag,
            )
            loss_dict["loss"].backward()
            optimizer.step()

            epoch_loss += float(loss_dict["loss"].item())
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

    adapter.eval()
    return {"status": "trained", "final_loss": loss_history[-1] if loss_history else None, "history": loss_history}


def evaluate_t0_model(
    model: NN_KNN_Model,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: T0Config,
    stats_store: CaseStatisticsStore | None = None,
    nominal_val: torch.Tensor | None = None,
    step: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Evaluate T0 NN-kNN model with pre/post adaptation metrics, flip analysis, and provenance tracking."""
    model.eval()
    model.to(device)
    num_samples = X_val.size(0)
    active_count = model.case_count()

    # Regression evaluation branch
    if cfg.task_type.lower() == "regression":
        y_targets = y_val.view(-1, 1).float().to(device)
        audit_res = None
        if stats_store is not None:
            audit_res = audit_regression_provenance_counterfactual(
                model=model,
                audit_queries=X_val,
                audit_targets=y_val,
                stats_store=stats_store,
                step=step,
                method=getattr(cfg, "regression_counterfactual_method", "counterfactual"),
                device=device,
            )

        with torch.no_grad():
            fwd_out = model(X_val.to(device))
            preds = fwd_out[0]
            pre_adapt = fwd_out[2] if len(fwd_out) > 2 else None
            if preds.dim() == 1:
                preds = preds.view(-1, 1)
            elif preds.dim() == 3:
                preds = preds.sum(dim=1)
            if pre_adapt is not None:
                if pre_adapt.dim() == 3:
                    pre_adapt = pre_adapt.sum(dim=1)
                elif pre_adapt.dim() == 1:
                    pre_adapt = pre_adapt.view(-1, 1)
            y_pred_pre = pre_adapt if pre_adapt is not None else preds
            pre_rmse = float(torch.sqrt(F.mse_loss(y_pred_pre, y_targets)).item())
            pre_mae = float(F.l1_loss(y_pred_pre, y_targets).item())
            post_rmse = float(torch.sqrt(F.mse_loss(preds, y_targets)).item())
            post_mae = float(F.l1_loss(preds, y_targets).item())

        return {
            "pre_accuracy": -pre_rmse,  # Higher is better proxy
            "post_accuracy": -post_rmse,
            "pre_rmse": pre_rmse,
            "post_rmse": post_rmse,
            "pre_mae": pre_mae,
            "post_mae": post_mae,
            "rmse_delta": post_rmse - pre_rmse,
            "active_cases": active_count,
            "total_flips": 0,
            "correct_flips": 0,
            "harmful_flips": 0,
            "net_flip_benefit": 0,
            "audit_result": audit_res,
        }

    # Classification evaluation branch
    y_idx = y_val.long().to(device) if y_val.dim() == 1 else y_val.argmax(dim=-1).to(device)
    batch_size = min(cfg.batch_size, num_samples)
    all_p0: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []

    with torch.no_grad():
        c_feats = (
            model.feature_extractor(model.cases[:active_count])
            if model.feature_extractor is not None
            else model.cases[:active_count]
        )
        for start in range(0, num_samples, batch_size):
            xb = X_val[start : start + batch_size].to(device)
            yb = y_idx[start : start + batch_size]
            q_feats = model.feature_extractor(xb) if model.feature_extractor is not None else xb

            q_exp = q_feats.unsqueeze(1).expand(-1, active_count, -1)
            c_exp = c_feats.unsqueeze(0).expand(xb.size(0), -1, -1)
            dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))
            scores = model.biases[:active_count].unsqueeze(0) - dists

            k_eff = min(cfg.top_k, active_count)
            top_scores, top_idx = torch.topk(scores, k=k_eff, dim=1)
            w = F.softmax(top_scores / cfg.tau, dim=1)

            top_case_feats = c_feats[top_idx]
            top_labels = model.labels[top_idx]

            nom_q = nominal_val[start : start + batch_size].to(device) if nominal_val is not None else None
            nom_ret = nominal_val[:active_count][top_idx].to(device) if nominal_val is not None else None

            Delta_z_q, p0_q, Delta_u_q = compute_classification_adaptation_inputs(
                query_features=q_feats,
                retrieved_features=top_case_feats,
                case_weights=w,
                retrieved_labels=top_labels,
                nominal_query=nom_q,
                nominal_retrieved=nom_ret,
            )
            all_p0.append(p0_q)

            adapter = getattr(model, "classification_adapter", None)
            if adapter is not None and cfg.classification_adapter_enabled:
                _, s_q = adapter(Delta_z_q, p0_q, Delta_u_q)
                all_scores.append(s_q)
            else:
                all_scores.append(p0_q)

            # Observe provenance if stats store is provided
            if stats_store is not None:
                top_case_ids = model.case_ids[top_idx].cpu().numpy()
                weights_np = w.cpu().numpy()
                pred_classes = top_labels.argmax(dim=-1).cpu().numpy()
                target_classes = yb.cpu().numpy()

                for b in range(xb.size(0)):
                    for k in range(k_eff):
                        cid = int(top_case_ids[b, k])
                        act = float(weights_np[b, k])
                        is_corr = bool(pred_classes[b, k] == target_classes[b])
                        stats_store.observe_retrieval([cid], [act], [is_corr], step=step)

    p0_all = torch.cat(all_p0, dim=0)
    scores_all = torch.cat(all_scores, dim=0)

    # Compute diagnostics and flip analysis
    flip_stats = ClassificationNNCDHAdapter.analyze_flips(p0_all, scores_all, y_idx)
    return {
        **flip_stats,
        "active_cases": active_count,
        "classification_adapter_enabled": cfg.classification_adapter_enabled,
        "p0": p0_all,
        "scores": scores_all,
    }


def run_t0_maintenance_step(
    model: NN_KNN_Model,
    stats_store: CaseStatisticsStore,
    archive_store: CaseArchiveStore,
    policy: CaseMaintenancePolicy,
    target_capacity: int,
    step: int = 0,
) -> tuple[int, list[MaintenanceAction]]:
    """Execute a single maintenance step: scores cases, archives evictions, and compacts model."""
    active_count = model.case_count()
    if active_count <= target_capacity:
        return active_count, []

    active_ids = model.active_case_ids().cpu().tolist()
    keep_ids, actions = policy.select_keep_case_ids(
        active_case_ids=active_ids,
        cases=model.cases,
        labels=model.labels,
        biases=model.biases,
        stats_store=stats_store,
        target_capacity=target_capacity,
        step=step,
    )

    keep_set = set(keep_ids)
    # Archive evicted cases
    for i, cid in enumerate(active_ids):
        if cid not in keep_set:
            st = stats_store.ensure(cid)
            archive_store.archive_case(
                case_id=cid,
                case_tensor=model.cases[i],
                label_tensor=model.labels[i],
                bias=float(model.biases[i].item()),
                stats=st,
                step=step,
                reason="policy_eviction",
            )

    # Compact model to keep_indices
    id_to_idx = {cid: idx for idx, cid in enumerate(active_ids)}
    keep_indices = [id_to_idx[cid] for cid in keep_ids if cid in id_to_idx]
    model.compact_cases(keep_indices)

    return model.case_count(), actions


def save_t0_checkpoint(
    path: str | Path,
    model: NN_KNN_Model,
    stats_store: CaseStatisticsStore,
    archive_store: CaseArchiveStore,
    cfg: T0Config,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Serialize full T0 CBR state including weights, active case IDs, provenance, archive, and config."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "active_case_count": model.case_count(),
        "case_ids": model.active_case_ids().cpu() if hasattr(model, "case_ids") else None,
        "stats_snapshot": {k: asdict(v) for k, v in stats_store.snapshot().items()},
        "archive_dict": {
            k: {
                "case_id": v["case_id"],
                "case_tensor": v["case_tensor"].cpu(),
                "label_tensor": v["label_tensor"].cpu(),
                "bias": v["bias"],
                "stats": asdict(v["stats"]),
                "archived_at_step": v["archived_at_step"],
                "eviction_reason": v["eviction_reason"],
            }
            for k, v in archive_store._archive.items()
        },
        "config": cfg.to_dict(),
        "extra_metadata": extra_metadata or {},
    }
    torch.save(state, p)


def load_t0_checkpoint(
    path: str | Path,
    X_sample: torch.Tensor,
    y_sample: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> tuple[NN_KNN_Model, CaseStatisticsStore, CaseArchiveStore, T0Config]:
    """Restore complete T0 CBR model, active cases, stats store, archive, and policy config."""
    p = Path(path)
    state = torch.load(p, map_location=device, weights_only=False)
    cfg_dict = state["config"]
    cfg = T0Config(**{k: v for k, v in cfg_dict.items() if k in T0Config.__dataclass_fields__})
    saved_cases = state["model_state_dict"].get("cases")
    saved_labels = state["model_state_dict"].get("labels")
    if saved_cases is not None and saved_labels is not None:
        init_X = torch.zeros_like(saved_cases)
        init_y = torch.zeros_like(saved_labels)
    else:
        init_X = X_sample
        init_y = y_sample
    model, stats_store, archive_store, _ = build_t0_model(init_X, init_y, cfg)
    model.load_state_dict(state["model_state_dict"])
    model.set_active_case_count(int(state["active_case_count"]))
    if state.get("case_ids") is not None and hasattr(model, "case_ids"):
        model.case_ids[: model.case_count()].copy_(state["case_ids"].to(model.case_ids.device))

    # Restore stats
    stats_store._stats.clear()
    for cid_int, st_dict in state.get("stats_snapshot", {}).items():
        st = CaseStatistics(**st_dict)
        stats_store._stats[int(st.case_id)] = st

    # Restore archive
    archive_store.clear()
    for cid_int, arc_data in state.get("archive_dict", {}).items():
        st_obj = CaseStatistics(**arc_data["stats"])
        archive_store.archive_case(
            case_id=arc_data["case_id"],
            case_tensor=arc_data["case_tensor"].to(device),
            label_tensor=arc_data["label_tensor"].to(device),
            bias=arc_data["bias"],
            stats=st_obj,
            step=arc_data["archived_at_step"],
            reason=arc_data["eviction_reason"],
        )
    return model, stats_store, archive_store, cfg
