from types import SimpleNamespace
import copy

import pytest
import torch

from model.nn_cdh import NNCDHAdapter
from model.common.core import CoreConfig, build_model, make_optimizer
from model.t1.reuse import ReuseConfig
from model.t1.adapted_retraining import continued_adapted_trial


def regression_fixture():
    X = torch.tensor([[-1., 0.], [-.6, .2], [-.2, -.1], [.2, .1], [.6, -.2], [1., 0.]])
    y = X[:, 0]*.7+X[:, 1]*.2
    cc = CoreConfig(task_type="regression", bias_init="manual", batch_size=3, seed=41)
    model = build_model(X, y, cc, None)
    opt = make_optimizer(model, cc)
    model.train()
    opt.zero_grad()
    loss = (model(X, exclude_identical=False)[0].flatten()-y).square().mean()
    loss.backward()
    opt.step()
    ad = NNCDHAdapter(2, 1, (8, 4))
    ad.adapt_net_pair.requires_grad_(False)
    aopt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=.001)
    r = model.retrieve(X, exclude_identical=False)
    aopt.zero_grad()
    final = ad.forward_aggregate(r["query_features"].detach(), r["case_features"].detach(),
        model.labels[r["case_indices"]].detach(), r["weights"].detach())
    (final.flatten()-y).square().mean().backward()
    aopt.step()
    rc = ReuseConfig(output_mode="nominal_residual_scores", loss="combined", lambda_diff=1.,
        lambda_cls=1., probability_mode="softmax", lr=.001)
    return model, opt, ad, aopt.state_dict(), SimpleNamespace(X_train=X, y_train=y), cc, rc


@pytest.mark.parametrize("epochs", [(1, 0), (0, 1), (1, 1)])
def test_regression_real_updates_only_declared_roles_and_no_pair_channel(epochs):
    model, opt, ad, ast, data, cc, rc = regression_fixture()
    original_model, original_ad, original_ast = copy.deepcopy(model.state_dict()), copy.deepcopy(ad.state_dict()), copy.deepcopy(ast)
    rng = torch.get_rng_state().clone()
    result = continued_adapted_trial(model, opt, ad, ast, data, cc, rc,
        retrieval_epochs=epochs[0], adapter_epochs=epochs[1], lr_scale=.1, drop_slot=2)
    m, ropt, a, aopt, info = result
    assert info["optimizer_updates"] == dict(retrieval=2*epochs[0], adapter=2*epochs[1])
    assert m.active_case_ids().tolist() == [0, 1, 3, 4, 5]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original_model.items())
    assert all(torch.equal(v, ad.state_dict()[k]) for k, v in original_ad.items())
    assert all(torch.equal(v, a.state_dict()[k]) for k, v in original_ad.items() if k.startswith("adapt_net_pair"))
    changed = any(not torch.equal(v, a.state_dict()[k]) for k, v in original_ad.items())
    assert changed == bool(epochs[1])
    assert {int(v["step"]) for v in aopt.state.values()} == {1+2*epochs[1]}
    assert {int(v["step"]) for v in ropt.state.values()} == {1+2*epochs[0]}
    assert all(torch.equal(original_ast["state"][i]["exp_avg"], ast["state"][i]["exp_avg"]) for i in ast["state"])
    assert info["objective_retrieval"] == "final_squared_error"


def test_regression_rejects_trainable_unused_pair_network():
    model, opt, ad, ast, data, cc, rc = regression_fixture()
    ad.adapt_net_pair.requires_grad_(True)
    with pytest.raises(ValueError, match="unused pair"):
        continued_adapted_trial(model, opt, ad, ast, data, cc, rc, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1)


def test_regression_initial_fit_restores_selected_adam_and_leaves_core_unchanged(monkeypatch):
    from model.t1 import reuse
    from dataclasses import replace
    model, opt, ad, ast, data, cc, rc = regression_fixture()
    data.task_type = "regression"
    data.X_val, data.y_val = data.X_train[:2]+.01, data.y_train[:2]
    rc = replace(rc, epochs=3, batch_size=3)
    original = copy.deepcopy(model.state_dict())
    real = reuse.regression_adapter_losses
    validations = iter([1., 3., 2.])
    def losses(final, p0, y, cfg):
        result = real(final, p0, y, cfg)
        if not final.requires_grad:
            result["loss"] = torch.tensor(next(validations))
        return result
    monkeypatch.setattr(reuse, "regression_adapter_losses", losses)
    fitted, info = reuse.train_regression_adapter(model, data, rc)
    assert info["best_epoch"] == 1 and info["epochs_run"] == 3
    assert {int(s["step"]) for s in info["optimizer_state"]["state"].values()} == {2}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    restored = torch.optim.Adam([p for p in fitted.parameters() if p.requires_grad], lr=rc.lr)
    restored.load_state_dict(copy.deepcopy(info["optimizer_state"]))
    r = model.retrieve(data.X_val, exclude_identical=False)
    restored.zero_grad()
    final = fitted.forward_aggregate(r["query_features"], r["case_features"], model.labels[r["case_indices"]], r["weights"])
    (final.flatten()-data.y_val).square().mean().backward()
    restored.step()
    assert {int(s["step"]) for s in restored.state.values()} == {3}
