from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter, compute_classification_adaptation_inputs
from model.nnknn_model import NN_KNN_Model
from model.t1_maintenance import (
    CaseArchiveStore,
    CaseMaintenancePolicy,
    CaseStatisticsStore,
    MaintenanceAction,
    get_maintenance_policy,
    write_maintenance_artifacts,
)
from model.t1_mcb import build_mcb_encoder_pair, compute_neighborhood_churn, compute_representation_drift
from model.t1_synchronization import (
    ComponentSynchronizer,
    FreeRadiusCalibrationResult,
    calibrate_free_correction_radius,
)


@dataclass
class T1Config:
    """Unified configuration surface for T1 Full-Cycle Neural CBR (Section 12)."""
    # 1. Maintenance & Retention
    case_maintenance_policy: str = "provenance_bias_coverage"
    case_capacity: int = 50
    case_maintenance_frequency: int = 100
    case_score_smoothing: float = 1.0
    case_trust_alpha: float = 0.6
    case_min_retrieval_count: int = 1
    case_min_activation_mass: float = 0.1
    case_bias_normalization: str = "within_cohort_percentile"
    case_redundancy_metric: str = "euclidean"
    case_min_per_class_or_action: int = 2
    case_archive_evictions: bool = True
    case_revision_enabled: bool = False

    # 2. Momentum Case Base (MCB-R)
    mcb_enabled: bool = False
    mcb_momentum: float = 0.999
    mcb_proj_dim: int = 64
    mcb_hidden_dim: int = 64
    mcb_normalize_embeddings: bool = True

    # 3. Classification Reuse Adapter
    classification_adapter_enabled: bool = False
    classification_adapter_architecture: str = "aggregate_label_conditioned"
    classification_adapter_input: str = "auto_by_representation_coverage"
    classification_adapter_output_mode: str = "nominal_residual_scores"  # or 'logit_residual'
    classification_adapter_freeze_retrieval_first: bool = True
    classification_adapter_lambda_diff: float = 1.0
    classification_adapter_lambda_cls: float = 1.0
    classification_adapter_lambda_adapt_cost: float = 0.0
    classification_adapter_lr: float = 1e-3
    classification_adapter_epochs: int = 100

    # 4. Component Synchronization & Free Correction
    component_sync_enabled: bool = False
    component_sync_free_correction_enabled: bool = True
    component_sync_schedule: str = "alternating"
    component_sync_lambda_pre: float = 1.0
    component_sync_lambda_post: float = 0.5
    component_sync_lambda_near: float = 0.01
    component_sync_lambda_delta: float = 0.5
    component_sync_lambda_small: float = 0.1

    # 5. Core Model Parameters
    top_k: int = 10
    tau: float = 1.0
    task_type: str = "classification"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def configuration_id(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:12]


def build_t1_model(
    cases: torch.Tensor,
    labels: torch.Tensor,
    cfg: T1Config,
    base_feature_extractor: nn.Module | None = None,
) -> tuple[NN_KNN_Model, CaseStatisticsStore, CaseArchiveStore, ClassificationNNCDHAdapter | None]:
    """Constructs the T1 NN-kNN model, case stores, MCB encoders, and reuse adapter."""
    n_cases = cases.size(0)
    in_dim = cases.view(n_cases, -1).size(-1)

    # 1. Setup MCB encoders if enabled
    feature_extractor = base_feature_extractor
    momentum_encoder = None
    if cfg.mcb_enabled:
        online_enc, mom_enc = build_mcb_encoder_pair(
            base_extractor=base_feature_extractor,
            in_dim=in_dim,
            hidden_dim=cfg.mcb_hidden_dim,
            proj_dim=cfg.mcb_proj_dim,
            normalize=cfg.mcb_normalize_embeddings,
        )
        feature_extractor = online_enc
        momentum_encoder = mom_enc

    # 2. Build core NN-kNN model
    model = NN_KNN_Model(
        cases=cases,
        labels=labels,
        feature_extractor=feature_extractor,
        momentum_encoder=momentum_encoder,
        task_type=cfg.task_type,
        top_k=cfg.top_k,
        tau=cfg.tau,
        mcb_momentum=cfg.mcb_momentum,
        mcb_normalize_embeddings=cfg.mcb_normalize_embeddings,
        explanation_mode=True,
    )

    # 3. Setup stores
    stats_store = CaseStatisticsStore()
    archive_store = CaseArchiveStore()

    labels_cpu = labels.detach().cpu()
    cohorts = (
        labels_cpu.argmax(dim=-1).tolist()
        if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1
        else labels_cpu.view(-1).long().tolist()
    )
    for i in range(n_cases):
        stats_store.register_case(
            case_id=i,
            initial_bias=float(model.biases[i].item()),
            cohort_id=cohorts[i],
            insertion_step=0,
            case_role=cfg.task_type,
        )

    # 4. Setup classification reuse adapter if enabled
    adapter: ClassificationNNCDHAdapter | None = None
    if cfg.classification_adapter_enabled and cfg.task_type == "classification":
        num_classes = labels.size(-1) if labels.dim() > 1 else int(labels.max().item()) + 1
        rep_dim = cfg.mcb_proj_dim if cfg.mcb_enabled else in_dim
        adapter = ClassificationNNCDHAdapter(
            feature_dim=rep_dim,
            num_classes=num_classes,
            output_mode=cfg.classification_adapter_output_mode,
        )

    return model, stats_store, archive_store, adapter


def run_t1_maintenance_step(
    model: NN_KNN_Model,
    stats_store: CaseStatisticsStore,
    archive_store: CaseArchiveStore,
    policy: CaseMaintenancePolicy,
    target_capacity: int,
    step: int = 0,
) -> tuple[int, list[MaintenanceAction]]:
    """Executes one maintenance checkpoint: scoring, selection, archiving, and compaction."""
    active_ids = model.active_case_ids().cpu().tolist()
    active_count = len(active_ids)
    if active_count <= target_capacity:
        return 0, [MaintenanceAction(cid, "keep", "within_capacity", step) for cid in active_ids]

    # Select cases to keep
    keep_ids, actions = policy.select_keep_case_ids(
        active_case_ids=active_ids,
        cases=model.cases,
        labels=model.labels,
        biases=model.biases,
        stats_store=stats_store,
        target_capacity=target_capacity,
        step=step,
    )

    keep_ids_set = set(keep_ids)
    active_id_to_idx = {cid: idx for idx, cid in enumerate(active_ids)}
    keep_indices = [active_id_to_idx[cid] for cid in keep_ids if cid in active_id_to_idx]

    # Archive evicted cases
    for act in actions:
        if act.action == "archive" and act.case_id in active_id_to_idx:
            idx = active_id_to_idx[act.case_id]
            st = stats_store.get(act.case_id)
            if st is not None:
                archive_store.archive_case(
                    case_id=act.case_id,
                    case_tensor=model.cases[idx],
                    label_tensor=model.labels[idx],
                    bias=float(model.biases[idx].item()),
                    stats=st,
                    step=step,
                    reason=act.reason,
                )

    # Compact active memory
    removed = model.compact_cases(keep_indices)
    return removed, actions


def evaluate_t1_model(
    model: NN_KNN_Model,
    test_loader: torch.utils.data.DataLoader,
    adapter: ClassificationNNCDHAdapter | None = None,
    log_retrievals: bool = False,
    run_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluates NN-kNN model with pre/post adaptation separation and contract event logging."""
    model.eval()
    if adapter is not None:
        adapter.eval()

    all_pre_preds: list[int] = []
    all_post_preds: list[int] = []
    all_targets: list[int] = []
    all_pre_probs: list[np.ndarray] = []
    all_post_probs: list[np.ndarray] = []
    retrieval_events: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (queries, targets) in enumerate(test_loader):
            queries = queries.to(model.cases.device)
            targets = targets.to(model.cases.device)

            if targets.dim() > 1 and targets.size(-1) > 1:
                target_cls = targets.argmax(dim=-1)
            else:
                target_cls = targets.view(-1).long()

            # Core forward pass
            res = model(queries)
            pre_probs = res[0]  # [B, C] class mass
            pre_preds = pre_probs.argmax(dim=-1)

            p0_q = pre_probs
            if adapter is not None:
                # Extract query & case representation for adapter input
                if model.feature_extractor is not None:
                    q_feat = model.feature_extractor(queries)
                    if getattr(model, "mcb_normalize_embeddings", False):
                        q_feat = F.normalize(q_feat, p=2, dim=-1)
                else:
                    q_feat = queries.view(queries.size(0), -1)

                active_cnt = model.case_count()
                case_idxs = torch.arange(active_cnt, device=queries.device)
                c_feat = model._extract_features(case_idxs)
                c_labels = model.labels[:active_cnt]

                q_exp = q_feat.unsqueeze(1).expand(-1, active_cnt, -1)
                c_exp = c_feat.unsqueeze(0).expand(queries.size(0), -1, -1)
                dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))
                scores = model.biases[:active_cnt].unsqueeze(0) - dists

                k_eff = min(model.top_k, active_cnt)
                top_scores, top_idx = torch.topk(scores, k=k_eff, dim=1)
                weights = F.softmax(top_scores / model.tau, dim=1)

                top_c_feats = c_feat[top_idx]
                top_c_labels = c_labels[top_idx]

                Delta_z_q, p0_calc, Delta_u_q = compute_classification_adaptation_inputs(
                    query_features=q_feat,
                    retrieved_features=top_c_feats,
                    case_weights=weights,
                    retrieved_labels=top_c_labels,
                )
                r_hat, s_q = adapter(Delta_z_q, p0_calc, Delta_u_q)
                post_preds = s_q.argmax(dim=-1)
                post_probs = F.softmax(s_q, dim=-1)
            else:
                post_preds = pre_preds
                post_probs = pre_probs

            all_pre_preds.extend(pre_preds.cpu().tolist())
            all_post_preds.extend(post_preds.cpu().tolist())
            all_targets.extend(target_cls.cpu().tolist())
            all_pre_probs.append(pre_probs.cpu().numpy())
            all_post_probs.append(post_probs.cpu().numpy())

            if log_retrievals:
                for q_i in range(queries.size(0)):
                    retrieval_events.append({
                        "retrieval_event_id": str(uuid.uuid4())[:8],
                        "project_run_id": run_id or "run_0",
                        "batch_idx": batch_idx,
                        "query_idx": q_i,
                        "pre_prediction": int(pre_preds[q_i].item()),
                        "post_prediction": int(post_preds[q_i].item()),
                        "target": int(target_cls[q_i].item()),
                        "p0_max": float(pre_probs[q_i].max().item()),
                    })

    y_true = np.array(all_targets)
    y_pre = np.array(all_pre_preds)
    y_post = np.array(all_post_preds)

    pre_acc = float(np.mean(y_pre == y_true))
    post_acc = float(np.mean(y_post == y_true))

    # Decision flip analysis (Section 11)
    flips = y_pre != y_post
    total_flips = int(np.sum(flips))
    correct_flips = int(np.sum(flips & (y_post == y_true)))
    harmful_flips = int(np.sum(flips & (y_pre == y_true)))
    net_benefit = correct_flips - harmful_flips

    metrics = {
        "pre_accuracy": pre_acc,
        "post_accuracy": post_acc,
        "accuracy_gain": post_acc - pre_acc,
        "total_flips": total_flips,
        "correct_flips": correct_flips,
        "harmful_flips": harmful_flips,
        "net_flip_benefit": net_benefit,
        "num_test_samples": len(y_true),
    }
    return metrics, retrieval_events


def train_leave_one_out_adapter(
    model: NN_KNN_Model,
    adapter: ClassificationNNCDHAdapter,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_diff: float = 1.0,
    lambda_cls: float = 1.0,
) -> list[dict[str, float]]:
    """Trains the reuse adapter with retrieval core frozen (T1 Section 5.5)."""
    model.eval()  # Freeze retrieval core
    adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    device = model.cases.device
    train_queries = train_queries.to(device)
    train_labels = train_labels.to(device)

    if train_labels.dim() > 1 and train_labels.size(-1) > 1:
        target_cls = train_labels.argmax(dim=-1)
    else:
        target_cls = train_labels.view(-1).long()

    active_cnt = model.case_count()
    if active_cnt <= 1:
        return []

    log_history: list[dict[str, float]] = []

    # Pre-extract representation with frozen encoder
    with torch.no_grad():
        if model.feature_extractor is not None:
            q_feat = model.feature_extractor(train_queries)
            if getattr(model, "mcb_normalize_embeddings", False):
                q_feat = F.normalize(q_feat, p=2, dim=-1)
        else:
            q_feat = train_queries.view(train_queries.size(0), -1)

        case_idxs = torch.arange(active_cnt, device=device)
        c_feat = model._extract_features(case_idxs)
        c_labels = model.labels[:active_cnt]

        q_exp = q_feat.unsqueeze(1).expand(-1, active_cnt, -1)
        c_exp = c_feat.unsqueeze(0).expand(train_queries.size(0), -1, -1)
        dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))
        scores = model.biases[:active_cnt].unsqueeze(0) - dists

        # Mask self-retrieval
        for i in range(min(train_queries.size(0), active_cnt)):
            scores[i, i] = float("-inf")

        k_eff = min(model.top_k, max(1, active_cnt - 1))
        top_scores, top_idx = torch.topk(scores, k=k_eff, dim=1)
        weights = F.softmax(top_scores / model.tau, dim=1)

        top_c_feats = c_feat[top_idx]
        top_c_labels = c_labels[top_idx]

        Delta_z_q, p0_q, Delta_u_q = compute_classification_adaptation_inputs(
            query_features=q_feat,
            retrieved_features=top_c_feats,
            case_weights=weights,
            retrieved_labels=top_c_labels,
        )

    for ep in range(epochs):
        optimizer.zero_grad()
        r_hat, s_q = adapter(Delta_z_q, p0_q, Delta_u_q)
        loss_dict = adapter.compute_loss(
            r_hat_q=r_hat,
            s_q=s_q,
            p0_q=p0_q,
            targets=target_cls,
            lambda_diff=lambda_diff,
            lambda_cls=lambda_cls,
        )
        total_loss = loss_dict["loss"]
        total_loss.backward()
        optimizer.step()

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log_history.append({
                "epoch": ep + 1,
                "loss_total": float(total_loss.item()),
                "loss_diff": float(loss_dict["loss_diff"].item()),
                "loss_cls": float(loss_dict["loss_cls"].item()),
            })

    return log_history



def write_t1_experiment_artifacts(
    output_dir: str | Path,
    cfg: T1Config,
    eval_metrics: dict[str, Any],
    stats_store: CaseStatisticsStore,
    archive_store: CaseArchiveStore,
    maintenance_log: list[MaintenanceAction],
    retrieval_events: list[dict[str, Any]] | None = None,
    calibration_result: FreeRadiusCalibrationResult | None = None,
    run_id: str | None = None,
) -> None:
    """Exports all required contract artifacts (Section 6 & 17)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"t1_{int(time.time())}_{str(uuid.uuid4())[:6]}"

    # 1. run_manifest.yaml
    manifest = {
        "run": {
            "id": rid,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_repository": "NN-KNN",
            "branch": "t1-neural-cbr",
            "seed": cfg.seed,
        },
        "model": {
            "task_type": cfg.task_type,
            "case_capacity": cfg.case_capacity,
            "top_k": cfg.top_k,
            "tau": cfg.tau,
        },
        "components": {
            "retrieve": "nnknn_activation",
            "reuse_adapter": cfg.classification_adapter_output_mode if cfg.classification_adapter_enabled else "none",
            "retain_policy": cfg.case_maintenance_policy,
            "revise_enabled": cfg.case_revision_enabled,
            "mcb_enabled": cfg.mcb_enabled,
            "component_synchronization": cfg.component_sync_enabled,
        },
        "metrics": eval_metrics,
    }
    with (out_path / "run_manifest.yaml").open("w", encoding="utf-8") as f:
        f.write("# T1 Experiment Manifest (SHARED_EXPERIMENT_AND_DATA_CONTRACT)\n")
        f.write(json.dumps(manifest, indent=2))

    # 2. summary.json
    summary = {
        "run_id": rid,
        "config": cfg.to_dict(),
        "metrics": eval_metrics,
        "calibration": calibration_result.to_dict() if calibration_result else None,
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 3. Case maintenance & statistics artifacts
    write_maintenance_artifacts(
        output_dir=out_path,
        stats_store=stats_store,
        actions_log=maintenance_log,
        archive_store=archive_store,
        run_id=rid,
    )

    # 4. Retrieval events log
    if retrieval_events:
        with (out_path / "retrieval_events.jsonl").open("w", encoding="utf-8") as f:
            for ev in retrieval_events:
                f.write(json.dumps(ev) + "\n")
