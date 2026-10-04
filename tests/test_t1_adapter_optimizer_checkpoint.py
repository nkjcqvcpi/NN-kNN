"""Continuation must use Adam moments from the selected state, not the last epoch."""
import copy
from types import SimpleNamespace

import torch

from model.t1.core import CoreConfig, build_model
from model.t1.reuse import ReuseConfig, train_classification_adapter
from model.t1.sync import SyncConfig, train_synchronized
import model.t1.reuse as reuse
import model.t1.sync as sync


def data_and_core():
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0, 0, 1, 1, 0, 1])
    cc = CoreConfig(task_type="classification", batch_size=3, top_k=3)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y, num_classes=2)
    return data, cc, build_model(X, y, cc, 2)


def capture_adam(monkeypatch):
    snapshots = {}
    original = torch.optim.Adam.step
    def step(opt, *args, **kwargs):
        result = original(opt, *args, **kwargs)
        snapshots.setdefault(opt, []).append(copy.deepcopy(opt.state_dict()))
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    return snapshots


def assert_state_equal(left, right):
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    assert left["state"]
    for key in left["state"]:
        for field, value in left["state"][key].items():
            assert torch.equal(value, right["state"][key][field])


def test_reuse_exports_selected_moments_and_resumes_exactly(monkeypatch, tmp_path):
    data, _, model = data_and_core()
    recorded = capture_adam(monkeypatch)
    original = reuse._adapter_losses
    validation_calls = [0]
    def losses(ad, r, s, p0, y, cfg):
        result = original(ad, r, s, p0, y, cfg)
        if len(y) == 6:
            validation_calls[0] += 1
            result["loss"] = torch.tensor(float(validation_calls[0]))
        return result
    monkeypatch.setattr(reuse, "_adapter_losses", losses)
    cfg = ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax",
                      epochs=3, patience=3, batch_size=3, hidden_dims=(8, 4))
    adapter, info = train_classification_adapter(model, data, cfg)
    assert info["epochs_run"] == 3 and info["optimizer_checkpoint_epoch"] == info["best_epoch"] == 1
    history = next(iter(recorded.values()))
    assert len(history) == 6
    assert_state_equal(info["optimizer_state"], history[1])
    assert all(int(v["step"]) == 2 for v in info["optimizer_state"]["state"].values())
    path = tmp_path / "adapter.pt"
    torch.save({"adapter": adapter.state_dict(), "optimizer": info["optimizer_state"]}, path)
    saved = torch.load(path, weights_only=True)
    a, b = copy.deepcopy(adapter), copy.deepcopy(adapter)
    opts = [torch.optim.Adam(ad.parameters(), lr=cfg.lr) for ad in (a, b)]
    for opt, state in zip(opts, (info["optimizer_state"], saved["optimizer"])):
        opt.load_state_dict(copy.deepcopy(state))
    b.load_state_dict(saved["adapter"])
    for ad, opt in zip((a, b), opts):
        opt.zero_grad()
        ad(torch.ones(2, 1), torch.full((2, 2), .5))[1].square().sum().backward()
        opt.step()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert_state_equal(opts[0].state_dict(), opts[1].state_dict())


def test_sync_exports_both_selected_optimizer_states(monkeypatch):
    data, cc, model = data_and_core()
    snapshots = capture_adam(monkeypatch)
    original = sync._forward_terms
    validations = [0]
    def terms(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(args[3]) == 6:
            validations[0] += 1
            result["L_post"] = torch.tensor(float(validations[0]))
        return result
    monkeypatch.setattr(sync, "_forward_terms", terms)
    cfg = SyncConfig("alternating_rr", 1., .5, .1, 1., 1., 0., epochs=6,
                     patience=6, block_epochs=2, hidden_dims=(8, 4))
    _, info = train_synchronized(model, data, cc, cfg, None, near_scale=1.)
    assert info["optimizer_checkpoint_epoch"] == info["best_epoch"] == 3
    histories = list(snapshots.values())
    core_history = next(h for h in histories if len(h) == 8)
    adapter_history = next(h for h in histories if len(h) == 4)
    assert_state_equal(info["retrieval_optimizer_state"], core_history[3])
    assert_state_equal(info["adapter_optimizer_state"], adapter_history[1])
