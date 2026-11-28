"""Frozen-retriever training and outcome routing for magnitude/direction reuse."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model.common.core import CoreConfig, build_model
from model.common.outcomes import final_prediction as core_final_prediction
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter
from model.t1.data import make_nominal_synthetic
from model.t1.nominal import encoded_queries
from model.t1.outcomes import final_prediction
from model.t1.reuse import (
    ReuseConfig, _adapter_losses, _classification_forward,
    evaluate_regression_reuse, evaluate_reuse, neighborhood_inputs,
    regression_adapter_losses, reuse_logits,
    train_classification_adapter, train_regression_adapter,
)


def config(**overrides):
    return replace(ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax",
        hidden_dims=(8, 4), epochs=3, batch_size=3, seed=17,
        adapter_design="magnitude_direction"), **overrides)


def fixture(task_type):
    X = torch.tensor([[-1., 0.], [-.6, .2], [-.2, -.1], [.2, .1], [.6, -.2], [1., 0.]])
    y = (X[:, 0] > 0).long() if task_type == "classification" else X[:, 0]*.7 + X[:, 1]*.2
    C = 2 if task_type == "classification" else None
    cc = CoreConfig(task_type=task_type, bias_init="manual", batch_size=3, top_k=3, seed=17)
    model = build_model(X, y, cc, C)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X+.03, y_val=y,
                           num_classes=C, task_type=task_type)
    return data, model


@pytest.mark.parametrize("overrides", [
    {"loss": "cls_only"}, {"loss": "diff_only"}, {"lambda_diff": 0.},
    {"lambda_cls": 0.}, {"lambda_diff": float("inf")},
    {"output_mode": "logit_residual"}, {"initial_scale": 0.},
])
def test_magnitude_reuse_rejects_missing_learning_signals(overrides):
    with pytest.raises(ValueError):
        config(**overrides).validate()


def test_neighborhood_cache_excludes_self_by_stable_id():
    data, model = fixture("classification")
    ids = model.active_case_ids()
    loo = neighborhood_inputs(model, data.X_train, exclude_identical=False,
                              query_case_ids=ids, include_statistics=True)
    same = neighborhood_inputs(model, data.X_train, exclude_identical=False,
                               include_statistics=True)
    assert torch.equal(loo["weights"].diagonal(), torch.zeros(len(ids)))
    assert (loo["statistics"][:, 0] > 0).all()
    torch.testing.assert_close(same["statistics"][:, 0], torch.zeros(len(ids)))
    assert not any(v.requires_grad for v in loo.values() if v is not None)


def test_regression_auxiliary_fits_raw_proposal_not_applied_correction():
    cfg = config()
    p0 = torch.tensor([[2.]])
    raw = torch.tensor([[.8]], requires_grad=True)
    magnitude = torch.tensor([[.1]], requires_grad=True)
    final = p0 + magnitude * raw.sign()
    losses = regression_adapter_losses(final, p0, torch.tensor([0.]), cfg, raw_proposal=raw)
    torch.testing.assert_close(losses["l_diff"], torch.tensor(2.8**2))
    losses["loss"].backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[5.6]]))
    assert magnitude.grad.item() > 0
    with pytest.raises(ValueError, match="raw proposal"):
        regression_adapter_losses(final, p0, torch.tensor([0.]), cfg)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_frozen_fit_preserves_core_and_outcomes_use_same_correction(task_type):
    data, model = fixture(task_type)
    before = copy.deepcopy(model.state_dict())
    cfg = config()
    train = train_classification_adapter if task_type == "classification" else train_regression_adapter
    adapter, info = train(model, data, cfg)
    assert info["adapter_design"] == "magnitude_direction"
    assert info["optimizer_checkpoint_epoch"] == info["best_epoch"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    r = model.retrieve(data.X_val, exclude_identical=False)
    diagnostics = {}
    pre, post, kind = final_prediction(model, data.X_val, r, adapter=adapter, diagnostics=diagnostics)
    torch.testing.assert_close((post-pre).norm(dim=1, keepdim=True), diagnostics["magnitude"])
    assert torch.isfinite(post).all()
    if task_type == "classification":
        nb = neighborhood_inputs(model, data.X_val, exclude_identical=False, include_statistics=True)
        raw, direct = _classification_forward(adapter, nb)
        losses = _adapter_losses(adapter, raw, direct, nb["p0"], data.y_val, cfg)
        torch.testing.assert_close(losses["l_diff"], F.mse_loss(raw, F.one_hot(data.y_val, 2).float()-nb["p0"]))
        torch.testing.assert_close(post, direct)
        torch.testing.assert_close(reuse_logits(model, adapter, data.X_val, cfg), cfg.score_scale*post)
        metrics = evaluate_reuse(model, adapter, data.X_val, data.y_val, cfg)
        assert metrics["accuracy_post"] == float((post.argmax(1) == data.y_val).float().mean())
        assert kind == "cross_entropy"
    else:
        metrics = evaluate_regression_reuse(model, adapter, data.X_val, data.y_val, success_tolerance=.5)
        assert metrics["rmse_post"] == pytest.approx(float((post.flatten()-data.y_val).square().mean().sqrt()))
        assert kind == "squared_error"


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_legacy_outcomes_remain_exact(task_type):
    data, model = fixture(task_type)
    adapter = ClassificationNNCDHAdapter(2, 2, hidden_dims=(8, 4)) if task_type == "classification" else NNCDHAdapter(2, 1, (8, 4))
    r = model.retrieve(data.X_val, exclude_identical=False)
    expected = core_final_prediction(model, data.X_val, r, adapter=adapter)
    actual = final_prediction(model, data.X_val, r, adapter=adapter)
    for left, right in zip(expected[:2], actual[:2]):
        assert torch.equal(left, right)
    assert expected[2] == actual[2]


def test_nominal_direction_uses_train_fitted_schema_and_explicit_queries():
    data = make_nominal_synthetic(9)
    fields = ({"name": "color", "covered_by_representation": False},
              {"name": "shape", "covered_by_representation": False},
              {"name": "size", "covered_by_representation": True})
    cfg = config(epochs=1, batch_size=60, nominal_fields=fields,
                 nominal_fields_covered_by_representation=False)
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification", top_k=3), data.num_classes)
    adapter, _ = train_classification_adapter(model, data, cfg)
    nominal = encoded_queries(adapter, data, "val")
    r = model.retrieve(data.X_val, exclude_identical=False)
    _, post, _ = final_prediction(model, data.X_val, r, adapter=adapter, query_nominal=nominal)
    nb = neighborhood_inputs(model, data.X_val, exclude_identical=False, X_nominal=nominal,
                             nominal_adapter=adapter, include_statistics=True)
    torch.testing.assert_close(post, _classification_forward(adapter, nb)[1])
    with pytest.raises(ValueError, match="explicit encoded"):
        final_prediction(model, data.X_val, r, adapter=adapter)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_retrieval_event_records_two_head_outputs(task_type):
    from model.t1.core import retrieval_events
    from model.t1.magnitude_direction import make_magnitude_direction_adapter

    data, model = fixture(task_type)
    adapter = make_magnitude_direction_adapter(task_type, 2, num_classes=data.num_classes, hidden_dims=(8, 4))
    events = retrieval_events(model, data.X_val, data.y_val, stream="test", run_id="two_head", adapter=adapter)
    r = model.retrieve(data.X_val, exclude_identical=False)
    _, expected, _ = final_prediction(model, data.X_val, r, adapter=adapter)
    for i, event in enumerate(events):
        assert event["retrieval_event_id"] == f"two_head-test-{i}"
        assert event["adapter_design"] == "magnitude_direction"
        assert len(event["neighborhood_statistics"]) == 4
        assert torch.tensor(event["direction"]).norm().item() == pytest.approx(1.)
        assert torch.tensor(event["predicted_residual"]).norm().item() == pytest.approx(event["magnitude"])
        prediction = int(expected[i].argmax()) if task_type == "classification" else float(expected[i, 0])
        assert event["final_prediction"] == prediction


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_legacy_retrieval_events_delegate_exactly(task_type):
    from model.common.core import retrieval_events as historical_events
    from model.t1.core import retrieval_events

    data, model = fixture(task_type)
    adapter = ClassificationNNCDHAdapter(2, 2, hidden_dims=(8, 4)) if task_type == "classification" else NNCDHAdapter(2, 1, (8, 4))
    kwargs = dict(stream="test", run_id="legacy", adapter=adapter)
    assert retrieval_events(model, data.X_val, data.y_val, **kwargs) == historical_events(model, data.X_val, data.y_val, **kwargs)


def test_removal_continuation_rejects_new_design_explicitly():
    from model.t1.adapted_retraining import continued_adapted_trial
    from model.t1.magnitude_direction import make_magnitude_direction_adapter

    data, model = fixture("regression")
    adapter = make_magnitude_direction_adapter("regression", 2, hidden_dims=(8, 4))
    with pytest.raises(ValueError, match="not supported by removal continuation"):
        continued_adapted_trial(model, object(), adapter, {}, data, CoreConfig(task_type="regression"),
                               config(), retrieval_epochs=1, adapter_epochs=1, lr_scale=1.)
