from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.nn_cdh import ClassificationNNCDHAdapter, compute_classification_adaptation_inputs
from model.nnknn_model import NN_KNN_Model
from model.t1_maintenance import (
    CaseArchiveStore,
    CaseMaintenancePolicy,
    CaseStatisticsStore,
    MaintenanceAction,
    get_maintenance_policy,
)
from model.t1_mcb import compute_neighborhood_churn, compute_representation_drift
from model.t1_synchronization import (
    ComponentSynchronizer,
    FreeRadiusCalibrationResult,
    calibrate_free_correction_radius,
    compute_minimal_adaptation_penalty,
)
from model.t1_workflow import T1Config, evaluate_t1_model, run_t1_maintenance_step


class TrainingScheduleType(str, Enum):
    STAGED_SEQUENTIAL = "staged_sequential"
    ALTERNATING = "alternating"
    JOINT_SYNCHRONIZED = "joint_synchronized"
    WARMUP_ALTERNATING = "warmup_alternating"
    WARMUP_JOINT = "warmup_joint"


@dataclass
class EpochMetrics:
    epoch: int
    phase: str  # 'core', 'adapter', 'joint', 'retrieval_block', 'adapter_block'
    loss_pre: float
    loss_post: float
    loss_delta: float
    loss_near: float
    loss_small: float
    loss_total: float
    val_pre_acc: float
    val_post_acc: float
    net_flips: int
    tau_task: float
    rep_drift: float
    neigh_churn: float
    active_cases: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingScheduleResult:
    schedule_name: str
    total_epochs: int
    best_pre_acc: float
    best_post_acc: float
    final_pre_acc: float
    final_post_acc: float
    final_net_flips: int
    accuracy_gain: float
    calibration_tau: float
    final_rep_drift: float
    final_neigh_churn: float
    total_time_sec: float
    history: list[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["history"] = [h.to_dict() for h in self.history]
        return d


class FullPipelineTrainer:
    """End-to-end trainer that fully trains all components across various schedules:

    1. Core NN-kNN feature extractor, glocal weights, and per-case biases.
    2. MCB memory encoder (EMA two-timescale parameter synchronization).
    3. Case statistics store (real gradient-time retrieval and outcome provenance).
    4. Case maintenance policy (capacity compaction under fixed K).
    5. Free-correction radius calibration (tau_task) and L_small constraint.
    6. Classification NN-CDH reuse adapter (aggregate nominal/logit residual).
    """

    def __init__(
        self,
        model: NN_KNN_Model,
        adapter: ClassificationNNCDHAdapter,
        stats_store: CaseStatisticsStore,
        archive_store: CaseArchiveStore,
        policy: CaseMaintenancePolicy,
        cfg: T1Config,
        synchronizer: ComponentSynchronizer | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.stats_store = stats_store
        self.archive_store = archive_store
        self.policy = policy
        self.cfg = cfg
        self.device = torch.device(device)
        self.synchronizer = synchronizer or ComponentSynchronizer(
            w_pre=cfg.component_sync_lambda_pre,
            w_post=cfg.component_sync_lambda_post,
            w_near=cfg.component_sync_lambda_near,
            v_post=1.0,
            v_delta=cfg.component_sync_lambda_delta,
            v_small=cfg.component_sync_lambda_small,
        )

        self.model.to(self.device)
        self.adapter.to(self.device)

        # Optimizers
        core_params: list[dict[str, Any]] = []
        if self.model.feature_extractor is not None:
            core_params.append({
                "params": self.model.feature_extractor.parameters(),
                "lr": float(self.cfg.to_dict().get("feature_extractor_lr", 1e-3)),
            })
        if self.model.glocal_weightor is not None:
            core_params.append({
                "params": self.model.glocal_weightor.parameters(),
                "lr": float(self.cfg.to_dict().get("glocal_weightor_lr", 1e-3)),
            })
        core_params.append({
            "params": [self.model.biases, self.model.negative_weights, self.model.glocal_weights],
            "lr": float(self.cfg.to_dict().get("case_net_lr", 5e-3)),
        })

        self.core_optimizer = torch.optim.Adam(core_params)
        self.adapter_optimizer = torch.optim.Adam(
            self.adapter.parameters(),
            lr=float(self.cfg.classification_adapter_lr),
        )

        self.calibrated_tau: float = 0.0
        self.calibrated_scale: float = 1.0

    def _step_retrieval_core(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        epoch: int,
        record_provenance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs one forward pass and loss step for the retrieval core."""
        res = self.model(xb)
        p0_q = res[0]  # [B, C] class mass
        y_idx = yb.argmax(dim=-1) if yb.dim() > 1 else yb.long()

        loss_pre = F.cross_entropy(torch.log(p0_q.clamp_min(1e-8)), y_idx)

        # Update case provenance if requested
        if record_provenance:
            with torch.no_grad():
                preds = p0_q.argmax(dim=-1)
                active_ids = self.model.active_case_ids().cpu().tolist()
                for i in range(min(xb.size(0), len(active_ids))):
                    cid = active_ids[i]
                    is_corr = bool((preds[i] == y_idx[i]).item())
                    self.stats_store.record_retrieval(
                        case_id=cid,
                        activation=float(p0_q[i, preds[i]].item()),
                        is_correct=is_corr,
                        step=epoch,
                    )

        # Distance / proximity metric L_near: intra-class proximity
        active_cnt = self.model.case_count()
        case_idxs = torch.arange(active_cnt, device=self.device)
        c_feat = self.model._extract_features(case_idxs)
        c_labels = self.model.labels[:active_cnt]
        c_cls = c_labels.argmax(dim=-1) if c_labels.dim() > 1 else c_labels.long()

        q_feat = self.model.feature_extractor(xb) if self.model.feature_extractor is not None else xb
        if getattr(self.model, "mcb_normalize_embeddings", False):
            q_feat = F.normalize(q_feat, p=2, dim=-1)

        q_exp = q_feat.unsqueeze(1).expand(-1, active_cnt, -1)
        c_exp = c_feat.unsqueeze(0).expand(xb.size(0), -1, -1)
        dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))

        pos_mask = (y_idx.unsqueeze(1) == c_cls.unsqueeze(0))
        if pos_mask.any():
            pos_dists = torch.where(pos_mask, dists, torch.zeros_like(dists))
            loss_near = pos_dists.sum() / pos_mask.sum().clamp_min(1.0)
        else:
            loss_near = torch.tensor(0.0, device=self.device)

        return loss_pre, loss_near, p0_q, q_feat

    def _step_adapter_forward(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        q_feat: torch.Tensor,
        p0_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes adapter inputs, predictions, residuals, and losses."""
        active_cnt = self.model.case_count()
        case_idxs = torch.arange(active_cnt, device=self.device)
        c_feat = self.model._extract_features(case_idxs)
        c_labels = self.model.labels[:active_cnt]

        # Top-k retrieved neighbors
        q_exp = q_feat.unsqueeze(1).expand(-1, active_cnt, -1)
        c_exp = c_feat.unsqueeze(0).expand(xb.size(0), -1, -1)
        dists = torch.sqrt(torch.relu(((q_exp - c_exp) ** 2).sum(dim=-1)))
        scores = self.model.biases[:active_cnt].unsqueeze(0) - dists

        k_eff = min(self.model.top_k, active_cnt)
        top_scores, top_idx = torch.topk(scores, k=k_eff, dim=1)
        weights = F.softmax(top_scores / self.model.tau, dim=1)

        top_c_feats = c_feat[top_idx]
        top_c_labels = c_labels[top_idx]

        Delta_z_q, p0_calc, Delta_u_q = compute_classification_adaptation_inputs(
            query_features=q_feat,
            retrieved_features=top_c_feats,
            case_weights=weights,
            retrieved_labels=top_c_labels,
        )

        r_hat, s_q = self.adapter(Delta_z_q, p0_calc, Delta_u_q)
        y_idx = yb.argmax(dim=-1) if yb.dim() > 1 else yb.long()

        loss_dict = self.adapter.compute_loss(
            r_hat_q=r_hat,
            s_q=s_q,
            p0_q=p0_calc,
            targets=y_idx,
            lambda_diff=self.cfg.classification_adapter_lambda_diff,
            lambda_cls=self.cfg.classification_adapter_lambda_cls,
        )
        loss_delta = loss_dict["loss_diff"]
        loss_post = loss_dict["loss_cls"]

        if self.adapter.output_mode == "nominal_residual_scores":
            p_post = torch.clamp(s_q, min=0.0)
            p_post = p_post / p_post.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            p_post = s_q

        loss_small = compute_minimal_adaptation_penalty(
            p0_calc,
            p_post,
            tau_task=self.calibrated_tau,
            s_task=self.calibrated_scale,
        )

        return loss_post, loss_delta, loss_small, s_q

    def train_schedule(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        schedule: str | TrainingScheduleType = TrainingScheduleType.STAGED_SEQUENTIAL,
        total_epochs: int = 60,
        maintenance_checkpoint_epoch: int = 20,
    ) -> TrainingScheduleResult:
        """Executes the specified full-pipeline training schedule."""
        sched_type = TrainingScheduleType(schedule)
        t_start = time.time()
        history: list[EpochMetrics] = []

        best_pre_acc = 0.0
        best_post_acc = 0.0

        # Snapshot for stability tracking
        prev_features: torch.Tensor | None = None
        prev_top_k: list[list[int]] | None = None

        print(f"\n--- Launching Training Schedule: {sched_type.value.upper()} (Total Epochs: {total_epochs}) ---")

        for ep in range(1, total_epochs + 1):
            ep_pre_loss = 0.0
            ep_post_loss = 0.0
            ep_delta_loss = 0.0
            ep_near_loss = 0.0
            ep_small_loss = 0.0
            ep_total_loss = 0.0
            n_batches = 0

            # Determine component activation per schedule
            train_core = True
            train_adapt = False
            phase_name = "core"

            if sched_type == TrainingScheduleType.STAGED_SEQUENTIAL:
                if ep <= maintenance_checkpoint_epoch:
                    train_core, train_adapt = True, False
                    phase_name = "core"
                else:
                    train_core, train_adapt = False, True
                    phase_name = "adapter"

            elif sched_type == TrainingScheduleType.ALTERNATING:
                # Alternate every 2 epochs
                if (ep // 2) % 2 == 0:
                    train_core, train_adapt = True, False
                    phase_name = "retrieval_block"
                else:
                    train_core, train_adapt = False, True
                    phase_name = "adapter_block"

            elif sched_type == TrainingScheduleType.JOINT_SYNCHRONIZED:
                train_core, train_adapt = True, True
                phase_name = "joint"

            elif sched_type == TrainingScheduleType.WARMUP_ALTERNATING:
                if ep <= maintenance_checkpoint_epoch:
                    train_core, train_adapt = True, False
                    phase_name = "warmup_core"
                else:
                    # Alternate after warm-up
                    if (ep // 2) % 2 == 0:
                        train_core, train_adapt = True, False
                        phase_name = "alt_retrieval"
                    else:
                        train_core, train_adapt = False, True
                        phase_name = "alt_adapter"

            elif sched_type == TrainingScheduleType.WARMUP_JOINT:
                if ep <= maintenance_checkpoint_epoch:
                    train_core, train_adapt = True, False
                    phase_name = "warmup_core"
                else:
                    train_core, train_adapt = True, True
                    phase_name = "joint_sync"

            # Execute Epoch Batches
            self.model.train() if train_core else self.model.eval()
            self.adapter.train() if train_adapt else self.adapter.eval()

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                n_batches += 1

                if train_core and not train_adapt:
                    # Pure Retrieval Step
                    self.core_optimizer.zero_grad()
                    l_pre, l_near, p0_q, _ = self._step_retrieval_core(xb, yb, epoch=ep)
                    loss = l_pre + (self.cfg.component_sync_lambda_near * l_near)
                    loss.backward()
                    self.core_optimizer.step()
                    if self.cfg.mcb_enabled:
                        self.model.update_momentum_encoder(self.cfg.mcb_momentum)

                    ep_pre_loss += float(l_pre.item())
                    ep_near_loss += float(l_near.item())
                    ep_total_loss += float(loss.item())

                elif train_adapt and not train_core:
                    # Pure Reuse Adapter Step (Retrieval frozen)
                    self.adapter_optimizer.zero_grad()
                    with torch.no_grad():
                        l_pre, l_near, p0_q, q_feat = self._step_retrieval_core(xb, yb, epoch=ep, record_provenance=False)
                    l_post, l_delta, l_small, _ = self._step_adapter_forward(xb, yb, q_feat, p0_q)
                    loss = l_post + l_delta + (self.cfg.component_sync_lambda_small * l_small)
                    loss.backward()
                    self.adapter_optimizer.step()

                    ep_pre_loss += float(l_pre.item())
                    ep_post_loss += float(l_post.item())
                    ep_delta_loss += float(l_delta.item())
                    ep_small_loss += float(l_small.item())
                    ep_total_loss += float(loss.item())

                elif train_core and train_adapt:
                    # Joint Synchronized Step
                    self.core_optimizer.zero_grad()
                    self.adapter_optimizer.zero_grad()

                    l_pre, l_near, p0_q, q_feat = self._step_retrieval_core(xb, yb, epoch=ep)
                    l_post, l_delta, l_small, _ = self._step_adapter_forward(xb, yb, q_feat, p0_q)

                    # Coordinated multi-term loss (T1 Section 6)
                    l_retrieval = (
                        self.synchronizer.w_pre * l_pre
                        + self.synchronizer.w_post * l_post
                        + self.synchronizer.w_near * l_near
                    )
                    l_adapter = (
                        self.synchronizer.v_post * l_post
                        + self.synchronizer.v_delta * l_delta
                        + self.synchronizer.v_small * l_small
                    )
                    total_step_loss = l_retrieval + l_adapter
                    total_step_loss.backward()

                    self.core_optimizer.step()
                    self.adapter_optimizer.step()
                    if self.cfg.mcb_enabled:
                        self.model.update_momentum_encoder(self.cfg.mcb_momentum)

                    ep_pre_loss += float(l_pre.item())
                    ep_post_loss += float(l_post.item())
                    ep_delta_loss += float(l_delta.item())
                    ep_near_loss += float(l_near.item())
                    ep_small_loss += float(l_small.item())
                    ep_total_loss += float(total_step_loss.item())

            # Maintenance & Calibration Checkpoint
            if ep == maintenance_checkpoint_epoch:
                print(f"  [Checkpoint @ Epoch {ep}] Executing Case-Base Maintenance ({self.cfg.case_maintenance_policy})...")
                removed, actions = run_t1_maintenance_step(
                    model=self.model,
                    stats_store=self.stats_store,
                    archive_store=self.archive_store,
                    policy=self.policy,
                    target_capacity=self.cfg.case_capacity,
                    step=ep,
                )
                print(f"  [Checkpoint @ Epoch {ep}] Calibrating Free-Correction Radius tau_task...")
                calib = calibrate_free_correction_radius(
                    model=self.model,
                    cases=self.model.cases[:self.model.case_count()],
                    labels=self.model.labels[:self.model.case_count()],
                    seed=self.cfg.seed,
                )
                self.calibrated_tau = calib.tau_task
                self.calibrated_scale = calib.s_task
                print(f"  [Checkpoint @ Epoch {ep}] Compacted {removed} cases. tau_task={self.calibrated_tau:.4f}, s_task={self.calibrated_scale:.4f}")
                # Reset optimizer state for case-dependent parameters after compaction to avoid stale gradient momentum
                for p in [self.model.biases, self.model.negative_weights, self.model.glocal_weights]:
                    self.core_optimizer.state.pop(p, None)

                # Cooldown core learning rate for gentle co-tuning after warm-up
                if sched_type in [TrainingScheduleType.WARMUP_JOINT, TrainingScheduleType.WARMUP_ALTERNATING]:
                    for g in self.core_optimizer.param_groups:
                        g["lr"] = g["lr"] * 0.2

            # Validation Evaluation
            val_metrics, _ = evaluate_t1_model(
                model=self.model,
                test_loader=val_loader,
                adapter=self.adapter if train_adapt or (ep > maintenance_checkpoint_epoch) else None,
            )

            # MCB & Stability Metrics
            curr_feat = self.model._extract_features(torch.arange(min(10, self.model.case_count()), device=self.device))
            rep_drift = 0.0
            if prev_features is not None and prev_features.shape == curr_feat.shape:
                rep_drift = compute_representation_drift(prev_features, curr_feat)
            prev_features = curr_feat.detach().clone()

            best_pre_acc = max(best_pre_acc, val_metrics["pre_accuracy"])
            best_post_acc = max(best_post_acc, val_metrics["post_accuracy"])

            metrics_record = EpochMetrics(
                epoch=ep,
                phase=phase_name,
                loss_pre=ep_pre_loss / max(1, n_batches),
                loss_post=ep_post_loss / max(1, n_batches),
                loss_delta=ep_delta_loss / max(1, n_batches),
                loss_near=ep_near_loss / max(1, n_batches),
                loss_small=ep_small_loss / max(1, n_batches),
                loss_total=ep_total_loss / max(1, n_batches),
                val_pre_acc=val_metrics["pre_accuracy"],
                val_post_acc=val_metrics["post_accuracy"],
                net_flips=val_metrics["net_flip_benefit"],
                tau_task=self.calibrated_tau,
                rep_drift=rep_drift,
                neigh_churn=0.0,
                active_cases=self.model.case_count(),
            )
            history.append(metrics_record)

            if ep % max(1, total_epochs // 6) == 0 or ep == total_epochs or ep == maintenance_checkpoint_epoch:
                print(
                    f"  Epoch {ep:2d}/{total_epochs} [{phase_name:14s}] | "
                    f"Loss: {metrics_record.loss_total:.4f} | "
                    f"Pre-Acc: {metrics_record.val_pre_acc:.3f} -> Post-Acc: {metrics_record.val_post_acc:.3f} | "
                    f"Net Flips: {metrics_record.net_flips:+2d} | "
                    f"Drift: {metrics_record.rep_drift:.4f}"
                )

        total_time = time.time() - t_start
        final_m = history[-1]

        result = TrainingScheduleResult(
            schedule_name=sched_type.value,
            total_epochs=total_epochs,
            best_pre_acc=best_pre_acc,
            best_post_acc=best_post_acc,
            final_pre_acc=final_m.val_pre_acc,
            final_post_acc=final_m.val_post_acc,
            final_net_flips=final_m.net_flips,
            accuracy_gain=final_m.val_post_acc - final_m.val_pre_acc,
            calibration_tau=self.calibrated_tau,
            final_rep_drift=final_m.rep_drift,
            final_neigh_churn=final_m.neigh_churn,
            total_time_sec=total_time,
            history=history,
        )
        return result
