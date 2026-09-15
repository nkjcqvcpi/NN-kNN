from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.t1_maintenance import get_maintenance_policy
from model.t1_scheduler import FullPipelineTrainer, TrainingScheduleType
from model.t1_workflow import T1Config, build_t1_model


def _create_mock_dataset(num_samples: int = 40, in_dim: int = 4, num_classes: int = 3, seed: int = 42):
    torch.manual_seed(seed)
    X = torch.randn(num_samples, in_dim)
    y_indices = torch.randint(0, num_classes, (num_samples,))
    y = F.one_hot(y_indices, num_classes=num_classes).float()
    return X, y


def test_full_pipeline_step_retrieval_and_adapter():
    """Verify single forward/backward steps for core retrieval and reuse adapter."""
    X, y = _create_mock_dataset(num_samples=20, in_dim=4, num_classes=3)
    cfg = T1Config(
        case_capacity=10,
        mcb_enabled=True,
        classification_adapter_enabled=True,
        classification_adapter_output_mode="nominal_residual_scores",
        seed=42,
    )
    model, stats, archive, adapter = build_t1_model(X, y, cfg)
    policy = get_maintenance_policy("provenance_bias_coverage", seed=42)

    trainer = FullPipelineTrainer(
        model=model,
        adapter=adapter,
        stats_store=stats,
        archive_store=archive,
        policy=policy,
        cfg=cfg,
    )

    xb, yb = X[:8], y[:8]
    l_pre, l_near, p0_q, q_feat = trainer._step_retrieval_core(xb, yb, epoch=1)
    assert l_pre.item() > 0.0
    assert l_near.item() >= 0.0
    assert p0_q.shape == (8, 3)

    l_post, l_delta, l_small, s_q = trainer._step_adapter_forward(xb, yb, q_feat, p0_q)
    assert l_post.item() > 0.0
    assert l_delta.item() >= 0.0
    assert l_small.item() >= 0.0
    assert s_q.shape == (8, 3)


def test_all_five_training_schedules_execute():
    """Verify that all 5 candidate schedules execute end-to-end and compact cases."""
    X, y = _create_mock_dataset(num_samples=30, in_dim=4, num_classes=3)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=True)
    val_loader = DataLoader(TensorDataset(X, y), batch_size=15, shuffle=False)

    schedules = [
        TrainingScheduleType.STAGED_SEQUENTIAL,
        TrainingScheduleType.ALTERNATING,
        TrainingScheduleType.JOINT_SYNCHRONIZED,
        TrainingScheduleType.WARMUP_ALTERNATING,
        TrainingScheduleType.WARMUP_JOINT,
    ]

    for sched in schedules:
        torch.manual_seed(42)
        cfg = T1Config(
            case_capacity=15,
            mcb_enabled=True,
            classification_adapter_enabled=True,
            classification_adapter_output_mode="nominal_residual_scores",
            seed=42,
        )
        model, stats, archive, adapter = build_t1_model(X, y, cfg)
        policy = get_maintenance_policy("provenance_bias_coverage", seed=42)

        trainer = FullPipelineTrainer(
            model=model,
            adapter=adapter,
            stats_store=stats,
            archive_store=archive,
            policy=policy,
            cfg=cfg,
        )

        res = trainer.train_schedule(
            train_loader=train_loader,
            val_loader=val_loader,
            schedule=sched,
            total_epochs=4,
            maintenance_checkpoint_epoch=2,
        )

        assert res.total_epochs == 4
        assert len(res.history) == 4
        assert res.history[-1].active_cases <= 15
        assert 0.0 <= res.final_pre_acc <= 1.0
        assert 0.0 <= res.final_post_acc <= 1.0


def test_staged_sequential_zero_drift_in_phase_two():
    """Verify that staged_sequential enforces zero representation drift once core is frozen."""
    X, y = _create_mock_dataset(num_samples=25, in_dim=4, num_classes=3)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    torch.manual_seed(42)
    cfg = T1Config(
        case_capacity=12,
        mcb_enabled=True,
        classification_adapter_enabled=True,
        classification_adapter_output_mode="nominal_residual_scores",
        seed=42,
    )
    model, stats, archive, adapter = build_t1_model(X, y, cfg)
    policy = get_maintenance_policy("provenance_bias_coverage", seed=42)

    trainer = FullPipelineTrainer(
        model=model,
        adapter=adapter,
        stats_store=stats,
        archive_store=archive,
        policy=policy,
        cfg=cfg,
    )

    res = trainer.train_schedule(
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=TrainingScheduleType.STAGED_SEQUENTIAL,
        total_epochs=6,
        maintenance_checkpoint_epoch=3,
    )

    # In Phase 2 (epochs 4, 5, 6), core feature extractor is frozen, so rep drift is 0.0
    for h in res.history[3:]:
        assert h.rep_drift == 0.0


if __name__ == "__main__":
    tests = [
        test_full_pipeline_step_retrieval_and_adapter,
        test_all_five_training_schedules_execute,
        test_staged_sequential_zero_drift_in_phase_two,
    ]
    passed = 0
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
        passed += 1
    print(f"\nAll {passed}/{len(tests)} scheduler verification tests passed successfully!")
