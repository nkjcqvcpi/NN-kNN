"""Leave-one-seed-out acceptance of selection rules and tuning blocks."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_t1_schedule_search import full_row, write_stage
from tools.t1_block_accept import block_main, paired, pool_rows, rules_main

SEEDS = [20, 21, 22, 23]
TASKS = [f"t{i}" for i in range(6)]


def stage(tmp_path, name, conds, rng, gain=None):
    """Conditions with validation errors that track their held-out test errors; ``gain`` lowers one condition's error."""
    rows = []
    for t in TASKS:
        for s in SEEDS:
            rows.append(full_row(t, "retrieval_only", s, .3, test_err=.3))
            for c, base in conds.items():
                e = base * (gain if gain and c == "new" else 1.0) * np.exp(rng.normal(0, .02))
                rows.append(full_row(t, c, s, e, val_loss=e, test_err=e))
    return write_stage(tmp_path / name, rows)


def args(**kw):
    return SimpleNamespace(seeds=SEEDS, reference="tuned::ref", prereg=None, exclude_condition=None,
                           conditions_from=None, **{"feasibility_pre": None, "tasks": None, **kw})


def test_paired_uses_task_clustered_se_and_requires_equal_cells():
    a = {(s, t): {"log_ratio": -0.1} for s in SEEDS for t in TASKS}
    b = {(s, t): {"log_ratio": 0.0 + 0.01 * (i % 3)} for i, (s, t) in enumerate(a)}
    d = paired(a, b)
    assert d["mean_difference"] < 0 and d["n_tasks"] == len(TASKS) and d["adopt"]
    with pytest.raises(SystemExit):
        paired(a, {k: v for k, v in list(b.items())[1:]})


def test_block_is_accepted_only_when_its_candidate_wins_held_out(tmp_path):
    rng = np.random.default_rng(0)
    pool = stage(tmp_path, "pool", {"ref": .30, "inc": .25}, rng)
    better = stage(tmp_path, "better", {"new": .18}, rng)
    worse = stage(tmp_path, "worse", {"new": .40}, rng)
    acc = block_main(args(pool=[str(pool)], block=[str(better)], rule=["err", "pertask"]))
    rej = block_main(args(pool=[str(pool)], block=[str(worse)], rule=["err", "pertask"]))
    assert acc["after_vs_before"]["adopt"] and acc["block_conditions_selected_held_out"] == ["tuned::new"]
    assert not rej["after_vs_before"]["adopt"] and rej["after_vs_before"]["mean_difference"] == pytest.approx(0)


def test_conditions_from_restricts_the_pool_and_first_stage_wins(tmp_path):
    rng = np.random.default_rng(1)
    first = stage(tmp_path, "first", {"ref": .30, "inc": .25}, rng)
    later = stage(tmp_path, "later", {"ref": .99, "inc": .99, "other": .1}, rng)
    rows = pool_rows([str(first), str(later)], conditions_from=[str(first)], seeds=SEEDS)
    conds = {r["condition"] for r in rows}
    assert "tuned__other" not in conds and all(r["val_accuracy_pre"] > .6 for r in rows if r["condition"] == "tuned__ref")


def test_rules_mode_keeps_the_current_rule_without_a_clear_gain(tmp_path):
    rng = np.random.default_rng(2)
    pool = stage(tmp_path, "pool", {"ref": .30, "inc": .25, "alt": .26}, rng)
    out = rules_main(args(pool=[str(pool)], current=["err", "global"]))
    assert set(out["rules"]) >= {"err/global", "err/pertask"} and out["decision"] in out["rules"]


def test_pertask_feasibility_excludes_a_candidate_only_where_it_degrades_retrieval(tmp_path):
    from tools.t1_block_accept import loso_cells
    rows = []
    for t in TASKS:
        for s in SEEDS:
            rows.append(full_row(t, "retrieval_only", s, .30, test_err=.30))
            rows.append(full_row(t, "ref", s, .30, val_loss=.3, test_err=.30))
            # "hide" wins on validation everywhere but wrecks retrieval on t0 only
            rows.append(full_row(t, "hide", s, .10, val_loss=.1, test_err=.12, pre_err=.90 if t == "t0" else .30))
    free = loso_cells(rows, SEEDS, "err", "pertask", "tuned::ref")
    guarded = loso_cells(rows, SEEDS, "err", "pertask", "tuned::ref", feasibility=1.02)
    assert {c["selected"] for c in free.values()} == {"tuned::hide"}
    assert all(c["selected"] == ("tuned::ref" if t == "t0" else "tuned::hide") for (s, t), c in guarded.items())
    every = loso_cells(rows, SEEDS, "err", "global", "tuned::ref", feasibility=1.02)
    assert {c["selected"] for c in every.values()} == {"tuned::ref"}  # the prereg's "on every task" form


def test_tasks_filter_limits_the_evaluation(tmp_path):
    rng = np.random.default_rng(3)
    pool = stage(tmp_path, "pool", {"ref": .30, "inc": .25}, rng)
    better = stage(tmp_path, "better", {"new": .18}, rng)
    out = block_main(args(pool=[str(pool)], block=[str(better)], rule=["err", "pertask"], tasks=["t0", "t1"]))
    assert out["tasks"] == ["t0", "t1"] and out["after_vs_before"]["n_tasks"] == 2


def test_split_mode_pools_classification_and_regression_cells(tmp_path):
    from tools.t1_block_accept import split_main
    rng = np.random.default_rng(4)

    def typed(name, conds, tt, gain=None):
        rows = []
        for t in ("a", "b", "c"):
            for s in SEEDS:
                rows.append(full_row(f"{tt[:3]}_{t}", "retrieval_only", s, .3, test_err=.3, task_type=tt))
                for c, base in conds.items():
                    e = base * (gain if gain and c.startswith("new") else 1.0) * np.exp(rng.normal(0, .02))
                    rows.append(full_row(f"{tt[:3]}_{t}", c, s, e, val_loss=e, test_err=e, task_type=tt))
        return write_stage(tmp_path / name, rows)
    cls_pool = typed("cp", {"ref__kappa10": .3, "inc__kappa10": .25}, "classification")
    cls_block = typed("cb", {"new__kappa10": .15}, "classification")
    reg_pool = typed("rp", {"ref": .3, "inc": .25}, "regression")
    reg_block = typed("rb", {"new": .15}, "regression")
    out = split_main(SimpleNamespace(cls_pool=[str(cls_pool)], cls_block=[str(cls_block)], cls_reference="tuned::ref__kappa10",
                                     reg_pool=[str(reg_pool)], reg_conditions_from=None, reg_block=[str(reg_block)],
                                     reference="tuned::ref", rule=["err", "pertask"], seeds=SEEDS, feasibility_pre=None, prereg=None))
    assert out["after_vs_before"]["n_tasks"] == 6 and out["after_vs_before"]["adopt"]
    assert set(out["block_conditions_selected_held_out"]) == {"tuned::new__kappa10", "tuned::new"}
