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
                           conditions_from=None, **kw)


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
