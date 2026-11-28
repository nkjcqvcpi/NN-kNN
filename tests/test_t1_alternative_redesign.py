import copy
import numpy as np
import pytest
import torch

from model.common import tracking
from model.t1 import records
from model.t1.alternative_adaptation import (
    VoteAttention, attention_tensors, attention_predict, retrieved_pairs,
    make_error_memory, error_memory_predict, predict_alternative, transport_predict,
)
from model.t1.alternative_training import fit_attention, fit_point_forests, make_folds
from test_t1_pairwise_redesign import fixture


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_attention_starts_at_retrieval_and_stays_in_support(kind):
    torch.manual_seed(17)
    context = fixture(kind)
    part = context["test"]
    model = VoteAttention(3, context["case_targets"].shape[1], dropout=.1).eval()
    np.testing.assert_allclose(attention_predict(model, context, part), part["p0"], atol=2e-7)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 5)
    prediction = attention_predict(model, context, part)
    support = context["case_targets"][part["indices"]]
    if kind == "regression":
        assert np.all(prediction >= support.min(1) - 2e-6)
        assert np.all(prediction <= support.max(1) + 2e-6)
    else:
        np.testing.assert_allclose(prediction.sum(1), 1.)
        assert np.all(prediction[support.sum(1) == 0] == 0)
    reverse = dict(part, indices=part["indices"][:, ::-1].copy(), weights=part["weights"][:, ::-1].copy())
    np.testing.assert_allclose(prediction, attention_predict(model, context, reverse), atol=1e-6, rtol=1e-6)


def test_attention_keeps_zero_weights_and_can_receive_gradients():
    context = fixture("regression")
    tensors = list(attention_tensors(context, context["train"]))
    tensors[3][:, 0] = 0
    tensors[3] /= tensors[3].sum(1, keepdim=True)
    model = VoteAttention(3, 1)
    prediction, weights = model(*tensors, return_weights=True)
    assert torch.count_nonzero(weights[:, 0]) == 0
    (prediction - torch.from_numpy(context["train"]["target"])[:, None]).square().mean().backward()
    assert torch.isfinite(model.net[-1].weight.grad).all()
    assert model.net[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_retrieved_only_pairs_and_error_memory_use_training_targets(kind):
    context = fixture(kind)
    pairs = retrieved_pairs(context)
    for q, i in zip(pairs["query_index"], pairs["case_index"]):
        assert i != q and i in context["train"]["indices"][q]
    mass = np.bincount(pairs["query_index"], weights=pairs["weight"])
    assert np.allclose(mass, mass[0])
    memory = make_error_memory(context)
    np.testing.assert_array_equal(memory["residual"], context["case_targets"].astype(np.float64) - context["train"]["p0"])
    changed = copy.deepcopy(context)
    del changed["test"]["target"]
    del changed["val"]["target"]
    np.testing.assert_array_equal(make_error_memory(changed)["z"], memory["z"])
    settings = dict(neighbors=8, multiplier=.25)
    before = error_memory_predict(memory, context, context["test"], settings)
    np.testing.assert_array_equal(before, error_memory_predict(memory, changed, changed["test"], settings))


def test_memory_can_exclude_own_stored_error():
    context = fixture("regression")
    model = make_error_memory(context)
    settings = dict(neighbors=1, multiplier=1.)
    training = context["train"]
    self_prediction = error_memory_predict(model, context, training, settings)
    np.testing.assert_allclose(self_prediction, context["case_targets"], atol=1e-7)
    loo = error_memory_predict(model, context, training, settings, query_ids=np.arange(len(model["z"])))
    assert not np.allclose(loo, self_prediction)


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("family", ["direct_trees", "anchor_transport", "vote_attention"])
def test_exact_recorded_recovery(tmp_path, monkeypatch, kind, family):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    context = fixture(kind)
    if family == "vote_attention":
        settings = dict(hidden=[8, 4], dropout=.1, score_cap=4., lr=.001, weight_decay=1e-4,
            grad_clip=5., batch_size=8, epochs=4, patience=20)
        def fit(run):
            return fit_attention(context, settings, seed=8, run=run)
        stop_step = 2
    else:
        settings = dict(n_estimators=8, tree_step=4, max_depth=4, max_features=.7, min_samples_leaf=2, folds=3)
        def fit(run):
            return fit_point_forests(context, settings, seed=8, run=run, transport=family == "anchor_transport")
        stop_step = 12 if family == "anchor_transport" else 4
    cfg = dict(kind=kind, family=family)
    with tracking.open_run(records.key("at", "full"), config=cfg) as run:
        expected, expected_info = fit(run)
    key = records.key("at", "resume")
    with pytest.raises(RuntimeError, match="interrupt"):
        with tracking.open_run(key, config=cfg) as run:
            save = run.save_resume
            def stop(*args, **kwargs):
                save(*args, **kwargs)
                if kwargs["step"] == stop_step:
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run, "save_resume", stop)
            fit(run)
    with tracking.open_run(key, config=cfg) as run:
        actual, actual_info = fit(run)
    assert expected_info == actual_info
    prediction = predict_alternative(actual, context, context["test"], family, settings)
    np.testing.assert_array_equal(prediction, predict_alternative(expected, context, context["test"], family, settings))
    no_targets = dict(context["test"])
    del no_targets["target"]
    np.testing.assert_array_equal(prediction, predict_alternative(actual, context, no_targets, family, settings))
    if kind == "classification":
        assert np.all(prediction >= 0)
        np.testing.assert_allclose(prediction.sum(1), 1.)
    if family == "anchor_transport":
        for case, fold in enumerate(actual["fold_of_case"]):
            assert case not in actual["training_indices"][fold]
    if family == "direct_trees":
        altered = dict(context["test"], p0=np.zeros_like(context["test"]["p0"]), indices=context["test"]["indices"][:, ::-1])
        np.testing.assert_array_equal(prediction, predict_alternative(actual, context, altered, family, settings))


def test_transport_recovers_affine_targets_with_consistent_potentials():
    context = fixture("regression")
    class Potential:
        def predict(self, z):
            return 2 + z[:, 0] - .3 * z[:, 1] + .6 * z[:, 2]
    splits, folds = make_folds(context, 7, 3)
    model = dict(models=[Potential()] * 3, fold_of_case=folds, anchor_predictions=context["case_targets"].copy())
    result = transport_predict(model, context, context["test"])
    np.testing.assert_allclose(result[:, 0], context["test"]["target"], atol=1e-6)


def test_completed_candidate_and_evaluation_are_reused(tmp_path, monkeypatch):
    from tools import t1_alternative_redesign as driver
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    context = fixture("regression")
    context["core_hash"] = "frozen"
    key = records.key("at", "context")
    with tracking.open_run(key, config={"toy": True}) as run:
        run.save_state("context", records.storable(context))
    unit = tracking.load_unit(key, config={"toy": True}, arrays=False)
    candidate = driver.run_candidate("at", "regression__toy", 3, "error_memory", "m0", unit, context)
    driver.evaluate_selected("at", "regression__toy", 3, "error_memory", [candidate], context)
    def forbid(*args, **kwargs):
        raise AssertionError("completed unit recomputed")
    monkeypatch.setattr(driver, "fit_alternative", forbid)
    monkeypatch.setattr(driver, "predict_alternative", forbid)
    driver.run_candidate("at", "regression__toy", 3, "error_memory", "m0", unit, context)
    driver.evaluate_selected("at", "regression__toy", 3, "error_memory", [candidate], context)


def test_merged_policy_excludes_direct_and_uses_only_validation():
    from tools.t1_alternative_redesign_summary import choose_all
    rows = [dict(method="direct_trees", val_error=0., test_error=0.),
        dict(method="retrieval", val_error=.1, test_error=.5),
        dict(method="anchor_transport", val_error=.1, test_error=.01)]
    assert choose_all(rows)["method"] == "retrieval"
    rows[2]["val_error"] = .09
    assert choose_all(rows)["method"] == "anchor_transport"
    rows[1]["test_error"], rows[2]["test_error"] = -100, 100
    assert choose_all(rows)["method"] == "anchor_transport"


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_support_bound_and_extended_behavior_aggregation(kind):
    from tools.t1_alternative_redesign_summary import support_diagnostics, verify_attention_support
    from tools.t1_pairwise_redesign_summary import behavior_totals, add_behavior
    context = fixture(kind)
    part = context["test"]
    diag = support_diagnostics(context, part)
    verify_attention_support(context, part, part["p0"])
    prediction = part["p0"].copy()
    if kind == "regression":
        target = part["target"].reshape(-1)
        values = context["case_targets"][part["indices"], 0]
        best = np.clip(target, values.min(1), values.max(1))
        assert diag["oracle_support_error"] == pytest.approx(np.sqrt(np.mean((target - best) ** 2)))
        prediction[:] = values.max() + 10
        with pytest.raises(AssertionError, match="escaped"):
            verify_attention_support(context, part, prediction)
    behavior, harms = [], []
    row = dict(task=kind + "__toy", task_type=kind, seed=1, method="vote_attention", unit="toy")
    add_behavior(row, dict(p0=part["p0"], final=part["p0"], target=part["target"]), behavior, harms)
    total, = behavior_totals(behavior, ("vote_attention",))
    assert total["n"] == len(part["target"])
