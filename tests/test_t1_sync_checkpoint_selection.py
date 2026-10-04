"""Maintenance must not erase a better checkpoint when it changes nothing."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.t1.calibration import calibrate_free_radius
from model.t1.core import CoreConfig, build_model
from model.t1.sync import SyncConfig, train_synchronized
import model.t1.sync as sync


def setup(monkeypatch):
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    cc = CoreConfig(task_type="regression", batch_size=3, top_k=3)
    model = build_model(X, y, cc, None)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y, num_classes=None)
    original = sync._forward_terms
    counter = [0]
    def terms(m, ad, x, target, *args, **kwargs):
        out = original(m, ad, x, target, *args, **kwargs)
        if len(x) == len(data.X_val):
            counter[0] += 1
            out["L_post"] = torch.tensor(float(counter[0]))
        return out
    monkeypatch.setattr(sync, "_forward_terms", terms)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    return model, data, cc, fr, counter


def config(schedule, patience=6):
    return SyncConfig(schedule, 1., .5, .1, 1., 1., .1, output_mode="aggregate_regression",
                       epochs=6, patience=patience, block_epochs=2, hidden_dims=(8, 4))


def test_noop_maintenance_preserves_best_checkpoint_and_model(monkeypatch):
    base, data, cc, fr, counter = setup(monkeypatch)
    rr = copy.deepcopy(base)
    ar, ir = train_synchronized(rr, data, cc, config("alternating_rr"), fr, near_scale=1.)
    counter[0] = 0
    rrr = copy.deepcopy(base)
    at, it = train_synchronized(rrr, data, cc, config("alternating_rrr"), fr, near_scale=1.,
                               maintenance_epochs={4}, maintenance_hook=lambda ep, m, opt, ad: {"n_before": m.case_count(), "n_after": m.case_count()})
    assert it["best_epoch"] == ir["best_epoch"] == 3
    assert it["history"][3]["maintenance_state_changed"] is False
    assert all(torch.equal(v, rrr.state_dict()[k]) for k, v in rr.state_dict().items())
    assert all(torch.equal(v, at.state_dict()[k]) for k, v in ar.state_dict().items())


def test_nonmaintenance_schedule_does_not_wait_for_unused_checkpoints(monkeypatch):
    m, data, cc, fr, _ = setup(monkeypatch)
    _, info = train_synchronized(m, data, cc, config("alternating_rr", patience=0), fr,
                                 near_scale=1., maintenance_epochs={5},
                                 maintenance_hook=lambda *args: pytest.fail("RR cannot maintain"))
    assert len(info["history"]) == 4


def test_actual_edit_at_same_count_resets_best_checkpoint(monkeypatch):
    m, data, cc, fr, _ = setup(monkeypatch)
    def edit(ep, model, opt, ad):
        with torch.no_grad():
            model.labels[0].add_(.1)
        return {"n_before": model.case_count(), "n_after": model.case_count()}
    _, info = train_synchronized(m, data, cc, config("alternating_rrr"), fr, near_scale=1.,
                                 maintenance_epochs={4}, maintenance_hook=edit)
    assert info["best_epoch"] == 4
    assert info["history"][3]["maintenance_state_changed"] is True
