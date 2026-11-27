"""Read recorded forests and independently check the matched RF comparison."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model.common import tracking
from model.t1 import records
from tools import t1_pair_random_forest as rf


def independent_error(prediction, target, kind):
    if kind == "classification":
        return float(np.mean(np.argmax(prediction, axis=1) != np.asarray(target).reshape(-1)))
    return float(np.sqrt(np.mean((np.asarray(prediction).reshape(-1) - np.asarray(target).reshape(-1)) ** 2)))


def audit(study, tasks, seeds):
    checks, values = Counter(), []
    def forbidden(*args, **kwargs):
        raise AssertionError("audit attempted fitting or inference instead of completed-unit reuse")
    saved_functions = rf.fit_forest, rf.predict
    rf.fit_forest = rf.predict = forbidden
    try:
        for task in tasks:
            kind = task.split("__")[0]
            for seed in seeds:
                context_unit, context = rf.a.source_context(task, seed)
                for family, reference_driver, reference_study, reference_family in (
                    (rf.FAMILIES[0], rf.p, "ar1", "pair_trees"),
                    (rf.FAMILIES[1], rf.a, "ar2", "local_pair_trees"),
                ):
                    candidates = []
                    for cid, settings in rf.CANDIDATES[family].items():
                        unit = rf.run_candidate(study, task, seed, family, cid, context_unit, context)
                        model = records.unpack(unit.load_state("model"))
                        expected = RandomForestClassifier if kind == "classification" else RandomForestRegressor
                        assert type(model) is expected
                        assert model.bootstrap is True and len(model.estimators_) == settings["n_estimators"]
                        assert all(tree.splitter == "best" for tree in model.estimators_)
                        for setting in ("max_depth", "max_features", "min_samples_leaf"):
                            assert getattr(model, setting) == settings[setting]
                        reference_settings = next(v for v in reference_driver.CANDIDATES[reference_family].values()
                            if v["min_samples_leaf"] == settings["min_samples_leaf"])
                        assert {k: v for k, v in settings.items() if k not in ("forest_kind", "bootstrap")} == reference_settings
                        checks["rf_estimators_and_matched_settings"] += 1
                        val = rf.p.numpy_state(unit.load_state("val_predictions"))
                        row = records.record(unit, "row")
                        assert np.isclose(independent_error(val["final"], val["target"], kind), row["val_error"], rtol=0, atol=1e-12)
                        assert not any(k.startswith("test_") for k in row)
                        checks["validation_errors"] += 1
                        candidates.append(unit)
                    selected = rf.evaluate_selected(study, task, seed, family, candidates, context)
                    rows = [records.record(unit, "row") for unit in candidates]
                    winner = min(rows, key=lambda r: (r["val_error"], r["val_task_loss"], r["candidate"]))
                    assert records.record(selected, "row")["candidate"] == winner["candidate"]
                    checks["validation_only_selections"] += 1
                    evaluation = ("eval_trees" if reference_driver is rf.p else "e_l")
                    reference = tracking.load_unit(reference_driver.key_for(reference_study, task, seed, evaluation), config=None, arrays=False)
                    assert reference is not None
                    for unit, method in ((selected, family), (reference, reference_family)):
                        stored = rf.p.numpy_state(unit.load_state("test_predictions"))
                        np.testing.assert_array_equal(stored["target"], context["test"]["target"])
                        np.testing.assert_array_equal(stored["p0"], context["test"]["p0"])
                        error = independent_error(stored["final"], stored["target"], kind)
                        assert np.isclose(error, records.record(unit, "row")["test_error"], rtol=0, atol=1e-12)
                        values.append(dict(task=task, task_type=kind, seed=seed, method=method, error=error))
                        checks["matched_test_errors"] += 1
                print(f"[audit] {task} s{seed}", flush=True)
    finally:
        rf.fit_forest, rf.predict = saved_functions
    task_means = []
    for task in tasks:
        for method in (*rf.FAMILIES, "pair_trees", "local_pair_trees"):
            group = [r["error"] for r in values if r["task"] == task and r["method"] == method]
            assert len(group) == len(seeds)
            task_means.append(dict(task=task, task_type=task.split("__")[0], method=method, error=float(np.mean(group))))
    family_means = [dict(task_type=kind, method=method,
        mean_error=float(np.mean([r["error"] for r in task_means if r["task_type"] == kind and r["method"] == method])))
        for kind in ("classification", "regression") if any(t.startswith(kind + "__") for t in tasks)
        for method in (*rf.FAMILIES, "pair_trees", "local_pair_trees")]
    payload = dict(study=study, sklearn_version=sklearn.__version__, checks=dict(checks),
        rows=values, task_means=task_means, family_means=family_means,
        reused_new_units=checks["validation_errors"] + checks["validation_only_selections"])
    out = rf.REPO / "results" / "adaptation_improvement" / study
    out.mkdir(parents=True, exist_ok=True)
    (out / "forest_audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rows", "task_means")}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--tasks", nargs="+", choices=rf.p.q.prior.TASKS, default=list(rf.p.q.prior.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(rf.p.q.prior.SEEDS))
    args = parser.parse_args()
    records.check_study(args.study)
    return audit(args.study, args.tasks, args.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
