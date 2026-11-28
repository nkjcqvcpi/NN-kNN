import copy
from dataclasses import replace
import pytest
import torch
from torch import nn

from model.common import tracking
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter
from model.t1 import records
from model.t1.adapter_regularization import add_hidden_dropout, dropout_description
from model.t1.query_aware import query_adapter_type
from model.t1.reuse import ReuseConfig, train_regression_adapter
from model.t1.sync import train_synchronized
from test_t1_training_schemes import fixture


def make_adapter(kind, design):
    typ = (ClassificationNNCDHAdapter if kind == "classification" else NNCDHAdapter) if design == "residual" else query_adapter_type(kind, design)
    adapter = typ(3, 2 if kind == "classification" else 1, hidden_dims=(8, 4))
    if kind == "regression":
        adapter.adapt_net_pair.requires_grad_(False)
    return adapter


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("design", ["residual", "query_aware", "query_aware_no_prediction"])
def test_dropout_preserves_weights_rng_eval_and_changes_training(kind, design):
    torch.manual_seed(19)
    baseline = make_adapter(kind, design)
    changed = copy.deepcopy(baseline)
    rng = torch.random.get_rng_state().clone()
    before = copy.deepcopy(changed.state_dict())
    add_hidden_dropout(changed, .2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert before.keys() == changed.state_dict().keys()
    assert all(torch.equal(v, changed.state_dict()[k]) for k, v in before.items())
    assert len(dropout_description(changed)) == 2
    network = lambda a: a.net if kind == "classification" else a.adapt_net_final.model
    x = torch.randn(64, network(changed)[0].in_features)
    baseline.eval(); changed.eval()
    assert torch.equal(network(baseline)(x), network(changed)(x))
    assert torch.equal(network(changed)(x), network(changed)(x))
    changed.train()
    assert not torch.equal(network(changed)(x), network(changed)(x))
    # Reloading the same named parameters into another configured adapter works.
    restored = add_hidden_dropout(make_adapter(kind, design), .2).eval()
    restored.load_state_dict(changed.state_dict())
    assert torch.equal(network(restored)(x), network(changed.eval())(x))
    zero = copy.deepcopy(baseline)
    assert add_hidden_dropout(zero, 0.) is zero
    assert dropout_description(zero) == []
    assert torch.equal(network(zero)(x), network(baseline)(x))


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("design", ["residual", "query_aware", "query_aware_no_prediction"])
@pytest.mark.parametrize("scheme", [None, "sequential", "alternating"])
def test_dropout_trains_under_every_scheme(kind, design, scheme):
    model, core, data, cfg = fixture(kind, design)
    initial = copy.deepcopy(model.state_dict())
    cfg = replace(cfg, adapter_dropout=.2, epochs=3)
    adapter, info = train_synchronized(model, data, core, cfg, None, near_scale=1.,
        comparison_scheme=scheme, dropout_placement="all_hidden")
    assert len(dropout_description(adapter)) == 2
    assert info["best_epoch"] >= 1
    assert all(torch.isfinite(p).all() for p in adapter.parameters())
    assert not adapter.training
    if scheme == "sequential":
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("scheme", [None, "sequential", "alternating"])
def test_dropout_rng_resume_matches_uninterrupted(tmp_path, monkeypatch, kind, scheme):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    model, core, data, cfg = fixture(kind, "query_aware")
    cfg = replace(cfg, adapter_dropout=.2)
    initial = copy.deepcopy(model.state_dict())
    def train(run):
        model.load_state_dict(initial); model._invalidate_case_cache()
        return train_synchronized(model, data, core, cfg, None, near_scale=1., run=run,
            comparison_scheme=scheme, dropout_placement="all_hidden")
    with tracking.open_run(records.key("dp", "full"), config={"kind":kind, "scheme":scheme}) as run:
        expected, info = train(run)
    expected_core = copy.deepcopy(model.state_dict())
    key = records.key("dp", "resume")
    with pytest.raises(RuntimeError, match="interrupt"):
        with tracking.open_run(key, config={"kind":kind, "scheme":scheme}) as run:
            save = run.save_resume
            def stop(*args, **kwargs):
                save(*args, **kwargs)
                if kwargs["step"] == 2:
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run, "save_resume", stop)
            train(run)
    with tracking.open_run(key, config={"kind":kind, "scheme":scheme}) as run:
        actual, resumed = train(run)
    assert info["history"] == resumed["history"]
    assert info["best_epoch"] == resumed["best_epoch"]
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected_core.items())


@pytest.mark.parametrize("design", ["residual", "query_aware", "query_aware_no_prediction"])
def test_frozen_regression_reuse_dropout_and_resume(tmp_path, monkeypatch, design):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    model, core, data, _ = fixture("regression", design)
    initial = copy.deepcopy(model.state_dict())
    cfg = ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax",
        hidden_dims=(8,4), epochs=4, patience=4, batch_size=3, seed=7, dropout=.2, adapter_design=design)
    def train(run):
        model.load_state_dict(initial); model._invalidate_case_cache()
        return train_regression_adapter(model, data, cfg, run=run, dropout_placement="all_hidden")
    with tracking.open_run(records.key("dp", "full"), config={"design":design}) as run:
        expected, info = train(run)
    assert len(dropout_description(expected)) == 2
    key = records.key("dp", "resume")
    with pytest.raises(RuntimeError, match="interrupt"):
        with tracking.open_run(key, config={"design":design}) as run:
            save = run.save_resume
            def stop(*args, **kwargs):
                save(*args, **kwargs)
                if kwargs["step"] == 2: raise RuntimeError("interrupt")
            monkeypatch.setattr(run, "save_resume", stop)
            train(run)
    with tracking.open_run(key, config={"design":design}) as run:
        actual, resumed = train(run)
    assert info["history"] == resumed["history"]
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())


def test_existing_classification_dropout_keeps_legacy_placement():
    model, core, data, cfg = fixture("classification", "residual")
    adapter, _ = train_synchronized(model, data, core, replace(cfg, adapter_dropout=.2), None, near_scale=1.)
    assert len(dropout_description(adapter)) == 1
    assert isinstance(adapter.net[3], nn.Dropout)


@pytest.mark.parametrize("p", [-.1, 1., float("nan")])
def test_invalid_dropout_is_rejected(p):
    with pytest.raises(ValueError, match="dropout"):
        add_hidden_dropout(make_adapter("regression", "residual"), p)


def test_dropout_selection_uses_validation_and_prefers_simpler_exact_ties():
    from tools.t1_adapter_dropout import choose
    rows=[dict(scheme=s,dropout=p,val_task_loss=l,val_error=e,test_error=t) for s,p,l,e,t in
          (("sequential",0.,.2,.1,0.),("alternating",.1,.1,.2,1.),("joint",.2,.1,.1,.9))]
    assert choose(rows)["dropout"] == .2
    rows[0]["test_error"], rows[2]["test_error"] = 100., 0.
    assert choose(rows)["dropout"] == .2
    for row in rows:
        row.update(val_task_loss=.1,val_error=.1)
    assert choose(list(reversed(rows)))["dropout"] == 0.
    rows[0]["dropout"] = .1
    assert choose(list(reversed(rows)))["scheme"] == "sequential"
