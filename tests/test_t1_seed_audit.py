"""Pre-registration seed audit: outer-split mapping, small test sets and split agreement."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from tools.t1_seed_audit import audit, load_prereg, main, outer_split_audit, role_violations, seed_roles

PREREG = "configs/t1/prereg_20261013.json"
SPLITS = {"test_frac": 0.2, "val_frac": 0.2, "maint_frac": 0.0, "reg_cohort_bins": 5}


def write_config(path, datasets, **splits):
    path.write_text(yaml.safe_dump({"splits": {**SPLITS, **splits}, "seeds": [20], "datasets": datasets}))
    return str(path)


def test_prereg_seeds_map_to_disjoint_outer_splits():
    prereg = load_prereg(PREREG)
    roles = seed_roles(prereg)
    assert role_violations(roles) == []
    table, bad = outer_split_audit("t", 10, 3, roles)
    split = {m["seed"]: tuple(m["split"]) for m in table}
    assert bad == [] and split[40] == (3, 1) and split[49] == (6, 1) and split[20] == (6, 2) and split[30] == (0, 0)
    assert outer_split_audit("t", 3, 3, roles)[1]  # 9 outer splits cannot hold 10 fresh evaluation seeds
    shifted = copy.deepcopy(prereg)
    shifted["seeds"]["evaluation"] = [40, 50, 70]  # 50 reuses seed 20's split; 70 repeats seed 40's
    bad = outer_split_audit("t", 10, 3, seed_roles(shifted))[1]
    assert any("seed 50" in v and "development seed 20" in v for v in bad)
    assert any("evaluation seeds 40 and 70 share" in v for v in bad)
    overlap = copy.deepcopy(prereg)
    overlap["seeds"]["development"].append(41)
    assert role_violations(seed_roles(overlap)) == ["seed 41 is listed under several roles: development, evaluation"]


def test_small_holdout_tasks_and_runner_grid_split_agreement(tmp_path):
    iris = [{"name": "iris", "task_type": "classification"}]
    nominal = {"name": "synthetic_nominal", "task_type": "classification"}  # fixed generator: streams are hashed
    report = audit([write_config(tmp_path / "ok.yaml", iris + [nominal])], PREREG, check_splits=True, split_seeds=[20, 40])
    assert report["violations"] == [] and report["small_tasks"] == ["classification__iris", "classification__synthetic_nominal"]
    assert [set(h["n_test"].values()) for h in report["holdout_tasks"]] == [{30}, {90}]
    assert [c["identical"] for c in report["split_checks"]] == [True] * 4
    # The grid caps the training stream from splits.max_train; the runner reads only the dataset entry.
    capped = write_config(tmp_path / "capped.yaml", iris, max_train=50)
    report = audit([capped], PREREG, check_splits=True, split_seeds=[20])
    assert report["violations"] == [f"{capped} classification__iris seed 20: runner and grid splits differ in ['train']"]
    assert main([capped, "--prereg", PREREG, "--check-splits", "--seeds", "20"]) == 1
    entry = write_config(tmp_path / "entry.yaml", [{**iris[0], "max_train": 50}], max_train=50)
    assert main([entry, "--prereg", PREREG, "--check-splits", "--seeds", "20", "--out", str(tmp_path / "a.json")]) == 0
    assert json.loads((tmp_path / "a.json").read_text())["split_checks"][0]["n"]["train"] == 50


def test_preregistered_training_cap_must_reach_the_dataset_entry_and_the_grid(tmp_path):
    from model.t1.data import make_splits
    prereg = load_prereg(PREREG)
    prereg["tasks"]["primary_legacy"] = [{"name": "iris", "task_type": "classification", "max_train": 50}]
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg))
    iris = {"name": "iris", "task_type": "classification"}
    report = audit([write_config(tmp_path / "c.yaml", [iris], max_train=50)], path)
    assert len(report["violations"]) == 1 and "dataset-entry max_train None != pre-registered 50" in report["violations"][0]
    # Cap only in the dataset entry: the runner trains on 50 rows, the grid's real call
    # (tools/t1_grid_search.py: make_splits(..., **study["splits"])) ignores the entry and trains on 90.
    entry_only = write_config(tmp_path / "e.yaml", [{**iris, "max_train": 50}])
    report = audit([entry_only], path, check_splits=True, split_seeds=[20])
    assert len(report["violations"]) == 2 and "splits.max_train None != pre-registered 50" in report["violations"][0]
    assert report["violations"][1].endswith("seed 20: runner and grid splits differ in ['train']")
    check = report["split_checks"][0]
    real = make_splits("iris", "classification", 20, **yaml.safe_load(Path(entry_only).read_text())["splits"])
    assert (check["n"]["train"], check["grid_n"]["train"]) == (50, len(real.meta["split_indices"]["train"])) == (50, 90)
    both = write_config(tmp_path / "b.yaml", [{**iris, "max_train": 50}], max_train=50)
    assert audit([both], path, check_splits=True, split_seeds=[20])["violations"] == []


def test_cached_tabarena_task_uses_the_audited_outer_split(tmp_path):
    openml = pytest.importorskip("openml")
    cache = Path(openml.config.get_cache_directory())
    if not (cache / "tasks" / "363614" / "datasplits.arff").exists() or not list((cache / "datasets" / "46906").glob("*.pq")):
        pytest.skip("OpenML task 363614 (anneal) is not cached")
    cfg = write_config(tmp_path / "anneal.yaml", [{"name": "tabarena_363614", "task_type": "classification"}])
    report = audit([cfg], PREREG, check_splits=True, split_seeds=[20, 40])
    task = report["openml_tasks"]["classification__tabarena_363614"]
    assert report["violations"] == [] and (task["repeats"], task["folds"]) == (10, 3)
    assert [(c["seed"], c["outer_split"], c["identical"]) for c in report["split_checks"]] == [(20, [6, 2], True), (40, [3, 1], True)]
    assert set(task["n_test"]) == set(range(40, 50)) and min(task["n_test"].values()) >= 100
