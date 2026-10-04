import copy
from types import SimpleNamespace

import pytest
import torch

from model.nn_cdh import NNCDHAdapter
from model.t1.calibration import calibrate_free_radius
from model.t1.candidates import MaintenanceReference, removal_influence
from model.t1.core import CoreConfig, build_model, train_retrieval
from model.t1.outcomes import final_prediction, frozen_evaluation, prediction_loss
from model.t1.reuse import evaluate_regression_reuse
from model.t1.sync import SyncConfig, _forward_terms, train_synchronized


def toy(task="regression"):
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2]) if task == "regression" else torch.tensor([0, 0, 0, 1, 1, 1])
    cc = CoreConfig(task_type=task, epochs=3, batch_size=3, top_k=3)
    m = build_model(X, y, cc, None if task == "regression" else 2)
    return m, cc, SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y, num_classes=None if task == "regression" else 2)


def sync_cfg(task, schedule):
    return SyncConfig(schedule, 1., .5, .1, 1., 1., .1,
                      output_mode="aggregate_regression" if task == "regression" else "nominal_residual_scores",
                      epochs=6, block_epochs=2, patience=6, hidden_dims=(8, 4))


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("schedule", ["independent", "alternating_rr", "alternating_rrr"])
def test_schedules_preserve_frozen_flags_and_aggregate_information_channel(task, schedule):
    m, cc, data = toy(task)
    flags = [p.requires_grad for p in m.parameters()]
    fr, _ = calibrate_free_radius(m, s_task=.5, snapshot_step=0)
    calls = []
    def hook(ep, model, optimizer, adapter):
        calls.append(ep)
        assert [p.requires_grad for p in model.parameters()] == flags
        return {"n_cases": model.case_count()}
    adapter, info = train_synchronized(m, data, cc, sync_cfg(task, schedule), fr, near_scale=1.,
                                      maintenance_epochs={3}, maintenance_hook=hook)
    assert [p.requires_grad for p in m.parameters()] == flags
    assert {h["phase"] for h in info["history"]} == {"retrieval", "adapter"}
    assert calls == ([3] if schedule == "alternating_rrr" else [])
    if task == "regression":
        assert all(not p.requires_grad for p in adapter.adapt_net_pair.parameters())
        metrics = evaluate_regression_reuse(m, adapter, data.X_val, data.y_val, success_tolerance=.5)
        assert metrics["rmse_post"] >= 0 and 0 <= metrics["success_rate_post"] <= 1
        T = _forward_terms(m, adapter, data.X_val, data.y_val, fr, output_mode="aggregate_regression", near_scale=1.)
        assert T["L_delta"].item() == pytest.approx(T["L_post"].item(), abs=1e-7)


def test_external_regression_adapter_is_actual_path_in_audit_and_removal():
    m, cc, data = toy()
    ad = NNCDHAdapter(1, 1, (8, 4))
    ref = MaintenanceReference(data.X_train, data.y_train, torch.arange(6), .5, ad)
    scores = removal_influence(m, ref)
    with frozen_evaluation(m, ad):
        for slot in (0, 3):
            mask = torch.ones(6, dtype=torch.bool)
            mask[slot] = False
            r = m.retrieve(data.X_train, exclude_identical=False, query_case_ids=ref.query_case_ids, case_mask=mask)
            _, p, kind = final_prediction(m, data.X_train, r, adapter=ad)
            loss = prediction_loss(p, data.y_train, kind).mean().item()
            assert scores["full"][slot] == pytest.approx(loss - scores["baseline_loss"], abs=1e-7)


def test_selected_core_restores_corresponding_adam_step(monkeypatch):
    import model.t1.core as core
    m, cc, data = toy()
    # Validation selects epoch 1 despite two further executed epochs.
    values = iter([1., 2., 3.])
    original_eval = core.evaluate
    def forced_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        result["loss_pre"] = next(values)
        return result
    monkeypatch.setattr(core, "evaluate", forced_eval)
    tr = train_retrieval(m, data.X_train, data.y_train, data.X_val, data.y_val, cc)
    assert tr.best_epoch == 1 and tr.epochs_run == 3
    assert tr.optimizer.state[m.biases]["step"].item() == 2  # two batches in selected epoch


def test_initial_checkpoint_restores_unadvanced_optimizer(monkeypatch):
    import model.t1.core as core
    m, cc, data = toy()
    before = copy.deepcopy(m.state_dict())
    values = iter([0., 1., 2.])
    original_eval = core.evaluate
    def forced_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        result["loss_pre"] = next(values)
        return result
    monkeypatch.setattr(core, "evaluate", forced_eval)
    tr = train_retrieval(m, data.X_train, data.y_train, data.X_val, data.y_val, cc, epochs=2, include_initial=True)
    assert tr.best_epoch == 0 and not tr.optimizer.state
    assert all(torch.equal(before[k], m.state_dict()[k]) for k in before)
