"""Core selection preserves best model/Adam state across no-op hooks."""
import copy
import pytest
import torch
from model.common.core import CoreConfig, build_model, train_retrieval
import model.common.core as core


def setup(monkeypatch, patience=6):
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    cfg = CoreConfig(task_type="regression", batch_size=3, top_k=3, epochs=6, patience=patience)
    m = build_model(X, y, cfg, None)
    original, calls = core.evaluate, [0]
    def evaluate(*args, **kwargs):
        out = original(*args, **kwargs)
        calls[0] += 1
        out["loss_pre"] = float(calls[0])
        return out
    monkeypatch.setattr(core, "evaluate", evaluate)
    return m, X, y, cfg, calls


@pytest.mark.parametrize("initial", [False, True])
def test_noop_hook_keeps_best_model_and_matching_optimizer(monkeypatch, initial):
    m, X, y, cfg, calls = setup(monkeypatch)
    plain = train_retrieval(copy.deepcopy(m), X, y, X+.01, y, cfg, include_initial=initial)
    calls[0] = 0
    hooked = train_retrieval(copy.deepcopy(m), X, y, X+.01, y, cfg, include_initial=initial,
        checkpoint_epochs={4}, checkpoint_hook=lambda *args: {"kept_all": True})
    assert hooked.best_epoch == plain.best_epoch == (0 if initial else 1)
    assert hooked.history[3]["checkpoint_state_changed"] is False
    for k, v in plain.model.state_dict().items():
        assert torch.equal(v, hooked.model.state_dict()[k])
    po, ho = plain.optimizer.state_dict(), hooked.optimizer.state_dict()
    assert po["param_groups"] == ho["param_groups"]
    assert set(po["state"]) == set(ho["state"])
    for pid, state in po["state"].items():
        for key, value in state.items():
            assert torch.equal(value, ho["state"][pid][key])


@pytest.mark.parametrize("hook,epochs", [(None, {5}), (lambda *args: {}, {99})])
def test_unexecutable_checkpoint_does_not_delay_early_stop(monkeypatch, hook, epochs):
    m, X, y, cfg, _ = setup(monkeypatch, patience=0)
    r = train_retrieval(m, X, y, X+.01, y, cfg, checkpoint_epochs=epochs, checkpoint_hook=hook)
    assert r.epochs_run == 2


def test_actual_label_edit_same_capacity_resets_selection(monkeypatch):
    m, X, y, cfg, _ = setup(monkeypatch)
    def edit(ep, model, opt):
        with torch.no_grad():
            model.labels[0].add_(.1)
        return {"n_cases": model.case_count()}
    r = train_retrieval(m, X, y, X+.01, y, cfg, checkpoint_epochs={4}, checkpoint_hook=edit)
    assert r.best_epoch == 4
    assert r.history[3]["checkpoint_state_changed"] is True
