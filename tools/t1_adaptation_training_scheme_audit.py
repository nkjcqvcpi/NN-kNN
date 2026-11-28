"""Audit recorded training schemes for the completed adapter redesign studies.

Read run units only. This does not train, predict, or modify existing metrics.
"""
from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.t1 import records
from tools import t1_pairwise_redesign as p
from tools import t1_alternative_redesign as a
from tools import t1_pair_random_forest as rf


def audit(study, tasks, seeds):
    rows, checks = [], Counter()
    for task in tasks:
        for seed in seeds:
            context_unit, context = a.source_context(task, seed)
            checks["contexts"] += 1
            for driver, source_study in ((p, "ar1"), (a, "ar2"), (rf, study)):
                for family in driver.FAMILIES:
                    evaluation = "eval_" + family.split("_")[-1] if driver is p else "e_" + driver.TAGS[family]
                    key = driver.key_for(source_study, task, seed, evaluation)
                    unit = tracking.load_unit(key, config=None, arrays=False)
                    if unit is None:
                        raise ValueError(f"Missing completed selected unit: {key.run_id}")
                    for name, expected in dict(protocol=driver.PROTOCOL, task=task, seed=seed,
                        family=family, role="selected").items():
                        if unit.config[name] != expected:
                            raise AssertionError(f"Selected unit {name} mismatch: {key.run_id}")
                    row = records.record(unit, "row")
                    if row["scheme"] != "sequential":
                        raise AssertionError(f"Unexpected recorded scheme: {key.run_id}")
                    if row["initial_core_hash"] != context["core_hash"] or row["selected_core_hash"] != context["core_hash"]:
                        raise AssertionError(f"Retrieval hash changed: {key.run_id}")
                    saved = p.numpy_state(unit.load_state("test_predictions"))
                    np.testing.assert_array_equal(saved["p0"], context["test"]["p0"])
                    np.testing.assert_array_equal(saved["target"], context["test"]["target"])
                    rows.append(dict(study=source_study, task=task, seed=seed, family=family,
                        recorded_scheme=row["scheme"], core_hash_matches=True, retrieval_prediction_matches=True,
                        context_unit=context_unit.info.run_id, selected_unit=key.run_id))
                    checks["selected_units"] += 1
                    checks["matching_core_hashes"] += 1
                    checks["matching_retrieval_predictions"] += 1
    groups = [dict(study=source, family=family, recorded_scheme=scheme, selected_units=n,
        matching_core_hashes=n, matching_retrieval_predictions=n)
        for (source, family, scheme), n in sorted(Counter((r["study"], r["family"], r["recorded_scheme"]) for r in rows).items())]
    out = REPO / "results" / "adaptation_improvement" / study
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("training_scheme_runs", rows), ("training_schemes", groups)):
        with (out / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    payload = dict(study=study, source_studies=["ar1", "ar2", study], checks=dict(checks),
        scope="recorded schemes, source core hashes and exact retrieval predictions; no new training or inference",
        interpretation="all redesign families use a sequential frozen-retrieval protocol; local ridge fits at inference, and vote attention changes only adapter aggregation weights",
        forest_joint_or_alternating_tested=False, groups=groups, rows=rows)
    (out / "training_scheme_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(out), checks=dict(checks), groups=groups), indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="Completed Random Forest study; ar1/ar2 references are reused")
    parser.add_argument("--tasks", nargs="+", choices=p.q.prior.TASKS, default=list(p.q.prior.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(p.q.prior.SEEDS))
    args = parser.parse_args()
    records.check_study(args.study)
    return audit(args.study, args.tasks, args.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
