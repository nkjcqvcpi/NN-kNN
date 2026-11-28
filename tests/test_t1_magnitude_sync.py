"""Both adaptation heads follow independent, alternating and joint schedules."""
import copy

import pytest
import torch
import torch.nn.functional as F

from model.common.core import CoreConfig, build_model
from model.t1.data import T1Data
from model.t1.magnitude_direction import make_magnitude_direction_adapter, neighborhood_statistics
from model.t1.outcomes import final_prediction
from model.t1.sync import SyncConfig, _forward_terms, frozen_parameters, train_synchronized


def fixture(task):
    x = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2]) if task == "regression" else torch.tensor([0, 0, 0, 1, 1, 1])
    data = T1Data("tiny", task, x, y, x + .01, y, x + .02, y,
                  num_classes=2 if task == "classification" else None)
    cc = CoreConfig(task_type=task, batch_size=3, top_k=3, bias_init_k=3)
    return build_model(x, y, cc, data.num_classes), cc, data


def config(task, schedule, **kw):
    return SyncConfig(schedule, 1., .5, .1, 1., 1., 0., epochs=4, patience=4,
        block_epochs=1, hidden_dims=(8, 4), adapter_design="magnitude_direction",
        output_mode="aggregate_regression" if task == "regression" else "nominal_residual_scores", **kw)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("schedule", ["independent", "alternating_rr", "joint_rr"])
def test_both_heads_train_and_selected_predictions_use_actual_scale(task, schedule, monkeypatch):
    import model.t1.sync as sync
    model, cc, data = fixture(task)
    original = sync.make_magnitude_direction_adapter
    initial = {}
    def capture(*args, **kwargs):
        adapter = original(*args, **kwargs)
        initial.update(copy.deepcopy(adapter.state_dict()))
        return adapter
    monkeypatch.setattr(sync, "make_magnitude_direction_adapter", capture)
    cfg = config(task, schedule, adapter_warmup_epochs=1 if schedule == "joint_rr" else 0)
    adapter, info = train_synchronized(model, data, cc, cfg, None, near_scale=1.)
    selected = adapter.state_dict()
    for head in ("direction_head", "scale_head"):
        assert any(not torch.equal(value, selected[name]) for name, value in initial.items() if head in name)
    assert info["best_epoch"] >= 1
    for epoch in info["history"]:
        assert epoch["predicted_magnitude"] == pytest.approx(epoch["correction_norm"], abs=1e-6)
        assert epoch["direction_norm"] == pytest.approx(1., abs=1e-6)
    r = model.retrieve(data.X_val, exclude_identical=False)
    diag = {}
    pre, final, _ = final_prediction(model, data.X_val, r, adapter=adapter, diagnostics=diag)
    assert torch.allclose((final-pre).norm(dim=1), diag["magnitude"].flatten(), atol=1e-6)
    if schedule == "joint_rr":
        first = info["history"][0]
        assert first["phase"] == "adapter"
        assert first["train_before_update"]["L_pre"] == pytest.approx(first["train_after_update"]["L_pre"], abs=1e-8)


def test_scalar_delta_loss_trains_raw_proposal_and_freezing_covers_both_heads():
    model, cc, data = fixture("regression")
    adapter = make_magnitude_direction_adapter("regression", 1, hidden_dims=(8, 4))
    r = model.retrieve(data.X_val, exclude_identical=False)
    labels = model.labels[r["case_indices"]].float()
    p0 = r["weights"] @ labels
    parts = adapter.forward_parts(r["query_features"]-r["weights"] @ r["case_features"], p0,
        neighborhood_statistics(r, labels, "regression"))
    terms = _forward_terms(model, adapter, data.X_val, data.y_val, None,
        output_mode="aggregate_regression", near_scale=1.)
    expected = F.mse_loss(parts.raw_proposal, data.y_val.reshape_as(p0)-p0)
    assert torch.allclose(terms["L_delta"], expected)
    flags = [p.requires_grad for p in adapter.parameters()]
    with frozen_parameters(adapter):
        assert not any(p.requires_grad for p in adapter.parameters())
    assert [p.requires_grad for p in adapter.parameters()] == flags


def test_missing_direction_learning_signal_is_rejected():
    cfg = config("regression", "joint_rr")
    cfg.v_delta = 0
    with pytest.raises(ValueError, match="raw-proposal"):
        cfg.validate("regression")
