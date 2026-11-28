import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.common import tracking
from model.common.core import make_optimizer
from model.t1 import records
from model.t1.forest_alternating import extract_context, online_prediction, train_retrieval_block, task_loss
from model.t1.pairwise_adaptation import predict_pairs, metrics
from model.t1.pairwise_training import fit_forest
from tools import t1_forest_alternating as driver
from test_t1_training_schemes import fixture


def setup(kind, family, tmp_path, monkeypatch):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    model, cc, data, _ = fixture(kind)
    context = extract_context(model, data, splits=("train", "val", "test"))
    context.update(core_hash=driver.p.s.state_hash(model.state_dict()), data_fingerprint="toy")
    settings = dict(n_estimators=8, tree_step=4, max_depth=3, max_features=.7, min_samples_leaf=2,
        forest_kind="random_forest" if "random_forest" in family else "extra_trees",
        bootstrap="random_forest" in family)
    ctx_key, core_key = records.key("af", "ctx"), records.key("af", "core")
    with tracking.open_run(ctx_key, config={"role": "ctx"}) as run:
        run.save_state("context", records.storable(context))
    with tracking.open_run(core_key, config={"role": "core"}) as run:
        run.save_state("objects", records.pack(dict(model=model, cc=cc,
            tr=SimpleNamespace(optimizer=make_optimizer(model, cc)))))
        records.save_rng(run)
    candidate_key = records.key("af", "base")
    with tracking.open_run(candidate_key, config={"role": "forest", "settings": settings}) as run:
        forest, _ = fit_forest(context, driver.training_pairs(context, family, 7), settings, seed=7, run=run)
        val = predict_pairs(forest, context, context["val"], family="pair_trees")
        test = predict_pairs(forest, context, context["test"], family="pair_trees")
        row = dict(unit=candidate_key.run_id, candidate="base",
            **{"val_" + k: v for k, v in metrics(val, context["val"]["target"], kind, context["val"]["p0"]).items()})
        run.save_state("model", records.pack(forest))
        for split, final in (("val", val), ("test", test)):
            part = context[split]
            run.save_state(split + "_predictions", records.storable(dict(final=final, p0=part["p0"], target=part["target"])))
        run.log_record("row", row)
    load = lambda k: tracking.load_unit(k, config=None, arrays=False)
    base = load(candidate_key)
    return model, cc, data, context, forest, load(core_key), load(ctx_key), dict(candidate=base, selected=base, settings=settings)


@pytest.mark.parametrize("kind", ("classification", "regression"))
@pytest.mark.parametrize("family", driver.FAMILIES)
def test_online_prediction_matches_existing_forest_and_pairs(kind, family, tmp_path, monkeypatch):
    model, cc, data, context, forest, *_ = setup(kind, family, tmp_path, monkeypatch)
    model.eval()
    with torch.no_grad():
        p0, final = online_prediction(model, data.X_val, forest, context)
    np.testing.assert_array_equal(p0.numpy(), context["val"]["p0"])
    np.testing.assert_allclose(final.numpy(), predict_pairs(forest, context, context["val"], family="pair_trees"), rtol=1e-13, atol=1e-13)
    changed = copy.deepcopy(context)
    changed["val"]["target"][:] = 999
    changed["test"]["target"][:] = -999
    with torch.no_grad():
        _, other = online_prediction(model, data.X_val, forest, changed)
    torch.testing.assert_close(final, other, rtol=0, atol=0)
    pairs = driver.training_pairs(context, family, 7)
    assert np.all(pairs["query_index"] != pairs["case_index"])
    assert driver.rf.pair_digest(pairs) == driver.rf.pair_digest(driver.training_pairs(changed, family, 7))


class FixedForest:
    classes_ = np.array([0, 1])
    def predict_proba(self, x):
        return np.broadcast_to(np.array([.3, .7]), (len(x), 2))
    def predict(self, x):
        return .2 * x[:, 0]


@pytest.mark.parametrize("kind", ("classification", "regression"))
def test_final_loss_has_real_retrieval_gradients_without_forest_gradient(kind):
    model, _, data, _ = fixture(kind)
    model.config["top_k"] = 5
    context = extract_context(model, data)
    model.train()
    _, final = online_prediction(model, data.X_train, FixedForest(), context,
        query_case_ids=torch.arange(len(data.X_train)))
    task_loss(final, data.y_train, kind).backward()
    assert model.biases.grad is not None
    assert torch.isfinite(model.biases.grad).all() and model.biases.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.feature_extractor.parameters())


@pytest.mark.parametrize("kind", ("classification", "regression"))
def test_retrieval_exact_resume_and_heldout_target_isolation(kind, tmp_path, monkeypatch):
    model, cc, data, context, forest, *_ = setup(kind, "pair_random_forest", tmp_path, monkeypatch)
    initial = copy.deepcopy(model.state_dict())
    def fit(run, changed_data=data):
        model.load_state_dict(initial)
        model._invalidate_case_cache()
        optimizer = make_optimizer(model, cc)
        history = train_retrieval_block(model, optimizer, changed_data, forest, context, cc,
            epochs=2, seed=9, run=run)
        return copy.deepcopy(model.state_dict()), optimizer.state_dict(), history
    with tracking.open_run(records.key("af", "full"), config={}) as run:
        expected, opt, history = fit(run)
    key = records.key("af", "resume")
    with pytest.raises(RuntimeError, match="interrupt"):
        with tracking.open_run(key, config={}) as run:
            save = run.save_resume
            def stop(*args, **kwargs):
                save(*args, **kwargs)
                if kwargs["step"] == 1:
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run, "save_resume", stop)
            fit(run)
    with tracking.open_run(key, config={}) as run:
        actual, actual_opt, actual_history = fit(run)
    assert actual_history == history
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    for i, state in opt["state"].items():
        for name in state:
            torch.testing.assert_close(state[name], actual_opt["state"][i][name], rtol=0, atol=0)
    changed = copy.deepcopy(data)
    # This tiny fixture deliberately shares y across splits: replace the
    # held-out tensors instead of mutating their shared training storage.
    changed.y_val = torch.full_like(changed.y_val, 999)
    changed.y_test = torch.full_like(changed.y_test, -999)
    with tracking.open_run(records.key("af", "isolated"), config={}) as run:
        isolated, _, _ = fit(run, changed)
    for name in expected:
        torch.testing.assert_close(expected[name], isolated[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind,family", (("classification", "pair_random_forest"), ("regression", "local_pair_trees")))
def test_recorded_cycle_and_completed_reuse(kind, family, tmp_path, monkeypatch):
    model, cc, data, context, forest, core, ctx, source = setup(kind, family, tmp_path, monkeypatch)
    monkeypatch.setitem(driver.PROTOCOL, "cycles", 2)
    monkeypatch.setitem(driver.PROTOCOL, "retrieval_epochs_per_cycle", 1)
    retrievals, refits = [], []
    previous_forest = source["candidate"]
    initial_core, initial_ctx = core, ctx
    for cycle in (1, 2):
        unit = driver.run_retrieval("af", kind + "__toy", 7, family, cycle, source, core, previous_forest, ctx, lambda: data)
        fitted = driver.run_refit("af", kind + "__toy", 7, family, cycle, source, unit)
        retrievals.append(unit)
        refits.append(fitted)
        core, ctx, previous_forest = unit, unit, fitted
        assert records.record(unit, "row")["core_hash"] != records.record(unit, "row")["previous_core_hash"]
        assert not any(k.startswith("test_") for k in records.record(fitted, "row"))
        assert "test" not in unit.load_state("context")
    result = driver.evaluate_selected("af", kind + "__toy", 7, family, source, retrievals, refits, context, lambda: data)
    chosen = driver.choose_checkpoint(driver.checkpoint_rows(source, refits))
    assert records.record(result, "row")["selected_cycle"] == chosen["cycle"]
    assert chosen["val_error"] <= records.record(source["candidate"], "row")["val_error"]
    from collections import Counter
    from tools import t1_forest_alternating_summary as summary
    monkeypatch.setattr(driver, "source_family", lambda *args: source)
    candidates, cycles, checks = [], [], Counter()
    audited, saved = summary.load_family("af", kind + "__toy", 7, family, initial_ctx, context,
        initial_core, candidates, cycles, checks)
    assert audited["selected_cycle"] == chosen["cycle"]
    assert checks["alternating_retrieval_states"] == 2
    assert checks["alternating_refit_validation_predictions"] == 2
    assert checks["alternating_selected_predictions_and_choices"] == 1
    assert len(candidates) == 2 and len(cycles) == 3
    def forbidden(*args, **kwargs):
        raise AssertionError("completed unit trained or predicted again")
    monkeypatch.setattr(driver, "train_retrieval_block", forbidden)
    monkeypatch.setattr(driver, "fit_forest", forbidden)
    monkeypatch.setattr(driver, "predict_pairs", forbidden)
    driver.run_retrieval("af", kind + "__toy", 7, family, 1, source, initial_core, source["candidate"], initial_ctx, forbidden)
    driver.run_refit("af", kind + "__toy", 7, family, 1, source, retrievals[0])
    driver.evaluate_selected("af", kind + "__toy", 7, family, source, retrievals, refits, context, forbidden)


def test_validation_choice_keeps_zero_and_never_uses_test():
    rows = [dict(cycle=0, val_error=.2, val_task_loss=.3, test_error=10.),
        dict(cycle=1, val_error=.2, val_task_loss=.3, test_error=0.)]
    assert driver.choose_checkpoint(rows)["cycle"] == 0
    rows[1]["val_error"] = .1
    assert driver.choose_checkpoint(rows)["cycle"] == 1
