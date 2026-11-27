import copy
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from model.common import tracking
from model.t1 import records
from model.t1.pairwise_training import fit_forest
from model.t1.pairwise_adaptation import forest_pair_predict, pair_training_data
from model.t1.alternative_adaptation import retrieved_pairs
from tools import t1_pair_random_forest as driver
from tools.t1_alternative_redesign_summary import choose_all
from test_t1_pairwise_redesign import fixture


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_rf_exact_resume_pair_reversal_and_target_isolation(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    context = fixture(kind)
    pairs = driver.training_pairs(context, "pair_random_forest", 7)
    settings = dict(n_estimators=8, tree_step=4, max_depth=4, max_features=.7,
        min_samples_leaf=2, forest_kind="random_forest", bootstrap=True)
    config = dict(kind=kind, settings=settings)
    with tracking.open_run(records.key("rf", "full"), config=config) as run:
        expected, info = fit_forest(context, pairs, settings, seed=3, run=run)
    assert type(expected) is (RandomForestClassifier if kind == "classification" else RandomForestRegressor)
    assert expected.bootstrap
    key = records.key("rf", "resume")
    with pytest.raises(RuntimeError, match="interrupt"):
        with tracking.open_run(key, config=config) as run:
            save = run.save_resume
            def stop(*args, **kwargs):
                save(*args, **kwargs)
                if kwargs["step"] == 4:
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run, "save_resume", stop)
            fit_forest(context, pairs, settings, seed=3, run=run)
    with tracking.open_run(key, config=config) as run:
        actual, resumed = fit_forest(context, pairs, settings, seed=3, run=run)
    assert info == resumed
    final = driver.predict(actual, context, context["test"])
    np.testing.assert_array_equal(final, driver.predict(expected, context, context["test"]))
    del context["test"]["target"]
    np.testing.assert_array_equal(final, driver.predict(actual, context, context["test"]))
    a, b = pairs["a"][:10], pairs["b"][:10]
    ab, ba = forest_pair_predict(actual, a, b, kind), forest_pair_predict(actual, b, a, kind)
    np.testing.assert_array_equal(ab, ba if kind == "classification" else -ba)
    if kind == "classification":
        assert np.all(final >= 0)
        np.testing.assert_allclose(final.sum(1), 1.)


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_rf_pairs_match_prior_recipes_and_ignore_heldout_targets(kind):
    context = fixture(kind)
    changed = copy.deepcopy(context)
    changed["val"]["target"][:] = 999
    changed["test"]["target"][:] = -999
    for family, expected in zip(driver.FAMILIES, (pair_training_data(context, seed=7, random_anchors=8), retrieved_pairs(context))):
        actual = driver.training_pairs(changed, family, 7)
        assert driver.pair_digest(expected) == driver.pair_digest(actual)
        assert (actual["query_index"] != actual["case_index"]).all()
        mass = np.bincount(actual["query_index"], weights=actual["weight"])
        np.testing.assert_allclose(mass, mass[0], rtol=1e-6)


@pytest.mark.parametrize("family", driver.FAMILIES)
def test_completed_rf_units_do_not_fit_or_predict_again(tmp_path, monkeypatch, family):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    context = fixture("regression")
    context["core_hash"] = "frozen"
    key = records.key("rf", "context")
    with tracking.open_run(key, config={"toy": True}) as run:
        run.save_state("context", records.storable(context))
    unit = tracking.load_unit(key, config={"toy": True}, arrays=False)
    cid = next(iter(driver.CANDIDATES[family]))
    settings = dict(driver.CANDIDATES[family][cid], n_estimators=8, tree_step=4)
    monkeypatch.setitem(driver.CANDIDATES[family], cid, settings)
    candidate = driver.run_candidate("rf", "regression__toy", 3, family, cid, unit, context)
    selected = driver.evaluate_selected("rf", "regression__toy", 3, family, [candidate], context)
    assert not any(k.startswith("test_") for k in records.record(candidate, "row"))
    assert records.record(selected, "row")["selected_core_hash"] == "frozen"
    def forbid(*args, **kwargs):
        raise AssertionError("completed unit recomputed")
    monkeypatch.setattr(driver, "fit_forest", forbid)
    monkeypatch.setattr(driver, "predict", forbid)
    driver.run_candidate("rf", "regression__toy", 3, family, cid, unit, context)
    driver.evaluate_selected("rf", "regression__toy", 3, family, [candidate], context)


def test_rf_only_changes_estimator_and_bootstrap_settings():
    for family, previous in zip(driver.FAMILIES, (driver.p.CANDIDATES["pair_trees"], driver.a.CANDIDATES["local_pair_trees"])):
        for settings, old in zip(driver.CANDIDATES[family].values(), previous.values()):
            assert {k: v for k, v in settings.items() if k not in ("forest_kind", "bootstrap")} == old
            assert settings["forest_kind"] == "random_forest" and settings["bootstrap"] is True


def test_expanded_policy_preserves_old_policy_and_never_uses_test():
    rows = [dict(method="direct_trees", val_error=0., test_error=0.),
        dict(method="pair_trees", val_error=.1, test_error=.5),
        dict(method="pair_random_forest", val_error=.09, test_error=10.)]
    assert choose_all(rows)["method"] == "pair_trees"
    assert choose_all(rows, include_rf=True)["method"] == "pair_random_forest"
    rows[2]["test_error"] = -100
    assert choose_all(rows, include_rf=True)["method"] == "pair_random_forest"
    rows[2]["val_error"] = .1
    assert choose_all(rows, include_rf=True)["method"] == "pair_trees"
