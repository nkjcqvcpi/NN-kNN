"""Condition grids and validation-only schedule selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy import stats

from tools.t1_run import expand_condition_grid, load_config
from tools.t1_schedule_search import condition_id, loso, pre_feasibility, rank_conditions, select, write_confirmation

STAGE = Path("configs/t1/schedule_search_20261011.yaml")
PREREG = "configs/t1/prereg_20261013.json"


def test_stage_grid_expands_to_unique_named_conditions():
    cfg = load_config(STAGE)
    names = [c["name"] for c in cfg["conditions"]]
    assert "condition_grid" not in cfg and len(names) == len(set(names)) == 26
    scratch = [c for c in cfg["conditions"] if c["name"].startswith("joint_scratch")]
    assert len(scratch) == 2 and all(c["core_pretrain"] is False and c["epochs"] == 300 for c in scratch)
    assert {"joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3",
            "alternating__block_epochs=1__retrieval_lr_scale=1.0", "alternating_default"} <= set(names)


def test_duplicate_grid_names_are_rejected():
    cfg = {"conditions": [{"name": "a__x=1"}], "condition_grid": [{"name": "a", "grid": {"x": [1]}}]}
    with pytest.raises(ValueError):
        expand_condition_grid(cfg)


def row(task, cond, seed, val_err, val_loss, task_type="classification", arm="tuned", **extra):
    r = {"dataset": task, "condition": f"{arm}__{cond}", "arm": arm, "seed": seed, "task_type": task_type,
         "val_L_post": val_loss, "test_accuracy_post": 0.0, "test_rmse_post": 99.0, **extra}
    if task_type == "classification":
        r["val_accuracy_post"] = 1 - val_err
    else:
        r["val_rmse_post"] = val_err
    if not cond.startswith(("nominal_residual", "aggregate_regression")):
        r["schedule"] = "x"
    return r


def test_reuse_rows_share_an_identity_across_task_types():
    a = row("t", "nominal_residual_scores_combined", 1, .1, .1, loss="combined")
    b = row("u", "aggregate_regression_combined", 1, .1, .1, task_type="regression", loss="combined")
    assert condition_id(a) == condition_id(b) == "tuned::frozen_reuse__combined"


def test_selection_uses_validation_error_then_loss_and_ignores_test():
    rows = []
    for seed in (20, 21):
        rows += [row("a", "joint", seed, .10, .50), row("a", "alt", seed, .10, .40),
                 row("b", "joint", seed, .20, .90, "regression"), row("b", "alt", seed, .30, .10, "regression"),
                 row("a", "retrieval_only", seed, .0, .0)]
    rows[0]["test_accuracy_post"] = 1.0  # test outcomes never change selection
    table, winners, glob, n = rank_conditions(rows)
    assert n == 2 and winners == {"a": "tuned::alt", "b": "tuned::joint"}
    assert {g["condition"] for g in glob} == {"tuned::alt", "tuned::joint"}
    assert all(g["mean_rank"] == 1.5 for g in glob)
    with pytest.raises(SystemExit):
        rank_conditions(rows[:-2])  # unequal seeds for one condition


def test_confirmation_config_carries_finalist_definitions(tmp_path):
    args = SimpleNamespace(stage_config=[str(STAGE)], confirm_template="configs/t1/schedule_search_confirm_20261011.yaml",
                           confirm_config=str(tmp_path / "confirm.yaml"), out=str(tmp_path))
    finalists = ["tuned::joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=0.3",
                 "tuned::frozen_reuse__combined", "tuned::alternating_default"]
    write_confirmation(args, finalists)
    conf = yaml.safe_load((tmp_path / "confirm.yaml").read_text())
    assert conf["seeds"] == [30, 31, 32, 33, 34] and conf["arms"] == [{"name": "tuned"}]
    assert [c["name"] for c in conf["conditions"]] == [finalists[0].split("::")[1], "alternating_default"]
    assert conf["conditions"][0]["retrieval_lr_scale"] == 0.3 and conf["reuse_conditions"]


def test_tuned_cores_reuse_the_grid_training_settings(tmp_path):
    import json
    from tools.t1_grid_search import nnknn_core_settings
    from tools.t1_schedule_search import prepare
    params = {"representation": "raw", "top_k": 10, "lr_case": 0.01, "glocal_fw_set_num": 4, "tau": 0.5}
    settings = nnknn_core_settings(params)
    assert settings["batch_size"] == 64 and settings["lr_glocal"] == .01 and settings["lr_feature"] == pytest.approx(.001)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"experiment": "x", "datasets": [{"name": "iris", "task_type": "classification"}]}))
    sel = tmp_path / "grid" / "classification__iris" / "nnknn"
    sel.mkdir(parents=True)
    (sel / "selection.json").write_text(json.dumps({"parameters": params, "test_stream_used": False}))
    out = tmp_path / "tuned.yaml"
    prepare(SimpleNamespace(template=str(template), grid=str(tmp_path / "grid"), out=str(out), experiment="y"))
    tuned = yaml.safe_load(out.read_text())
    assert tuned["core_by_dataset"]["classification__iris"] == settings and tuned["experiment"] == "y"


# ----------------------------------------------------------------------------- selection options and LOSO

def full_row(task, cond, seed, val_err, val_loss=.5, test_err=.5, pre_err=None, task_type="classification",
             n_val=50, n_test=40):
    """Validation and test metrics as the runner writes them; retrieval-only rows carry no stream sizes."""
    r = row(task, cond, seed, val_err, val_loss, task_type)
    m = (lambda e: 1 - e) if task_type == "classification" else (lambda e: e)
    kind = "accuracy" if task_type == "classification" else "rmse"
    if cond == "retrieval_only":
        r.update({f"val_{kind}_pre": m(val_err), f"test_{kind}_pre": m(test_err)})
    else:
        r.update({f"val_{kind}_pre": m(val_err if pre_err is None else pre_err), f"test_{kind}_post": m(test_err),
                  "val_n": n_val, "test_n": n_test})
    return r


def write_stage(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs__x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def sel_args(out, dirs, **kw):
    return SimpleNamespace(exp_dir=[str(d) for d in dirs], out=str(out), finalists=1, reference=None,
                           confirm_template=None, exclude_condition=None, **kw)


def test_select_restricts_seeds_and_refuses_prereg_seeds(tmp_path):
    rows = [full_row("a", c, s, e) for c, e in (("p", .1), ("q", .2)) for s in (20, 21)]
    stage = write_stage(tmp_path / "stage", rows + [full_row("a", "p", 22, .9), full_row("a", "q", 22, .0)])

    def run(name, dirs=(stage,), **kw):
        select(sel_args(tmp_path / name, dirs, **kw))
        return json.loads((tmp_path / name / "selection.json").read_text())

    plain = run("plain")
    assert plain["global_winner"] == "tuned::q" and not {"seeds_requested", "criterion", "prereg"} & set(plain)
    dev = run("dev", seeds=[20, 21], prereg=PREREG)
    assert dev["global_winner"] == "tuned::p" and dev["development_seeds"] == [20, 21]
    assert dev["prereg"]["refused_roles"] == ["evaluation", "evaluation_small_tasks", "retired", "reserved"]
    with pytest.raises(SystemExit, match="no rows for requested seeds"):
        run("absent", seeds=[20, 26])
    with pytest.raises(SystemExit, match=r"40 \(evaluation\)"):
        run("asked", seeds=[20, 40], prereg=PREREG)
    late = write_stage(tmp_path / "late", [full_row("a", "p", 30, .1)])
    with pytest.raises(SystemExit, match=r"30 \(retired\)"):
        run("loaded", dirs=(stage, late), seeds=[20, 21], prereg=PREREG)


def test_criteria_order_by_error_loss_or_mean_seed_rank():
    errs = {"p": (0., .25, .25), "q": (.3, .2, .2), "r": (.5, .5, .5)}
    loss = {"p": .9, "q": .5, "r": .05}
    rows = [full_row("a", c, s, e, loss[c]) for c, es in errs.items() for s, e in zip((20, 21, 22), es)]
    assert [rank_conditions(rows, criterion=k)[1]["a"] for k in ("err", "loss", "rank")] == ["tuned::p", "tuned::r", "tuned::q"]
    table = rank_conditions(rows, criterion="rank")[0]
    assert {t["condition"]: t["mean_seed_rank"] for t in table} == pytest.approx(
        {"tuned::p": 5 / 3, "tuned::q": 4 / 3, "tuned::r": 3.0})
    assert "mean_seed_rank" not in rank_conditions(rows)[0][0]
    unpaired = rows[:-1] + [full_row("a", "r", 21, .5, .05)]  # equal counts, but r lacks seed 22
    rank_conditions(unpaired)
    with pytest.raises(SystemExit, match="every seed"):
        rank_conditions(unpaired, criterion="rank")


def test_feasibility_excludes_candidates_whose_pre_adaptation_error_drifts(tmp_path):
    rows = []
    for s in (20, 21):
        rows += [full_row("a", "retrieval_only", s, .10), full_row("a", "p", s, .02, pre_err=.14),
                 full_row("a", "q", s, .05, pre_err=.12), full_row("a", "z", s, .08, pre_err=.10),
                 full_row("b", "retrieval_only", s, .5, task_type="regression")]
        rows += [full_row("b", c, s, .4, pre_err=.5, task_type="regression", n_val=100) for c in ("p", "q", "z")]
    bad = pre_feasibility(rows, 1.02)  # task a (n_val 50: 5, 7, 6, 5 errors): limit 1.02 * 0.10 + 1/50
    assert list(bad) == ["tuned::p"] and [w["task"] for w in bad["tuned::p"]] == ["a"]
    assert bad["tuned::p"][0]["limit"] == pytest.approx(.122) and bad["tuned::p"][0]["n_val"] == 50
    select(sel_args(tmp_path / "sel", [write_stage(tmp_path / "stage", rows)], feasibility_pre=1.02))
    sel = json.loads((tmp_path / "sel" / "selection.json").read_text())
    assert sel["global_winner"] == "tuned::q" and "tuned::p" not in [g["condition"] for g in sel["global_ranking"]]
    assert sel["feasibility_pre"]["ineligible"]["tuned::p"][0]["val_pre_error"] == pytest.approx(.14)
    import csv
    with (tmp_path / "sel" / "validation_table.csv").open() as fh:
        table = list(csv.DictReader(fh))
    assert [t["rank"] for t in table if t["condition"] == "tuned::p"] == ["", ""]
    with pytest.raises(SystemExit, match="retrieval-only"):
        pre_feasibility([r for r in rows if not (r["condition"].endswith("retrieval_only") and r["seed"] == 21)], 1.02)


def test_feasibility_counts_classification_errors_in_whole_validation_cases():
    """One extra validation error over a perfect retrieval-only lies exactly on the bound and stays feasible."""
    acc32 = lambda k, n: float(np.float32((n - k) / n))  # noqa: E731  accuracy as the runner stores it
    assert 1 - acc32(1, 30) > 1 / 30 and 1 - acc32(1, 36) > 1 / 36  # an exact float comparison refuses the bound
    rows = []
    for s in (20, 21, 22):
        for task, n in (("iris", 30), ("wine", 36)):
            rows.append(full_row(task, "retrieval_only", s, 0.))
            for cond, k in (("one", 1), ("two", 2)):
                rows.append({**full_row(task, cond, s, 0., n_val=n), "val_accuracy_pre": acc32(k, n)})
    bad = pre_feasibility(rows, 1.02)
    assert list(bad) == ["tuned::two"] and [w["task"] for w in bad["tuned::two"]] == ["iris", "wine"]
    assert bad["tuned::two"][0]["val_pre_error"] == pytest.approx(2 / 30) and bad["tuned::two"][0]["limit"] == pytest.approx(1 / 30)
    rows[1]["val_accuracy_pre"] = 0.95  # 1.5 errors in 30 cases: val_n does not fit the accuracy
    with pytest.raises(SystemExit, match="does not fit"):
        pre_feasibility(rows, 1.02)


def loso_rows():
    """Seeds 20, 21, 22. Task a's validation depends on the seed, so leaving out a seed changes its
    winner: seeds 21+22 pick q, 20+22 pick p, 20+21 pick q, while all three seeds would pick p
    (and each seed alone p, q, p). Task b's validation always prefers q, then ref, then p.
    On task a's test stream q beats p."""
    test = {("a", "p"): (.10, .20, .30), ("a", "q"): (.05, .05, .05), ("a", "ref"): (.20, .20, .20),
            ("b", "p"): (.6, .6, .6), ("b", "q"): (.4, .5, .6), ("b", "ref"): (.5, .5, .5)}
    val = {("a", "p"): (.0, .5, .1), ("a", "q"): (.4, .0, .3), ("a", "ref"): (.6, .6, .6),
           ("b", "p"): (.5,) * 3, ("b", "q"): (.3,) * 3, ("b", "ref"): (.4,) * 3}
    return [full_row(t, c, s, v, test_err=e, task_type="classification" if t == "a" else "regression")
            for (t, c), es in test.items() for s, v, e in zip((20, 21, 22), val[t, c], es)]


def loso_args(dirs, out, **kw):
    base = dict(exp_dir=[str(d) for d in dirs], seeds=[20, 21, 22], rules=["err"], scope=["global"], reference=None,
                incumbent=None, challengers=None, prereg=None, exclude_condition=None, out=str(out))
    return SimpleNamespace(**{**base, **kw})


def test_loso_selects_on_other_seeds_validation_and_scores_held_out_test(tmp_path):
    stage, out = write_stage(tmp_path / "stage", loso_rows()), tmp_path / "loso.json"
    loso(loso_args([stage], out, rules=["err", "rank"], scope=["global", "pertask"], reference="tuned::ref"))
    rep = json.loads(out.read_text())
    res = {(r["rule"], r["scope"]): r for r in rep["results"]}
    eps = .5 / 40  # classification task a; regression task b has eps 0
    a = lambda e: np.log((e + eps) / (.2 + eps))  # noqa: E731
    b = [np.log(x / .5) for x in (.4, .5, .6)]
    # Global always picks q (task b's winner keeps the lower mean rank); per task, task a follows the
    # other seeds' validation: q, p, q for held-out seeds 20, 21, 22, scored on that seed's test row.
    glob, per = [a(.05)] * 3 + b, [a(.05), a(.20), a(.05)] + b

    def check(r, cells):
        v = np.array(cells)
        assert r["n"] == 6 and r["mean_log_ratio"] == pytest.approx(v.mean())
        assert r["se"] == pytest.approx(v.std(ddof=1) / np.sqrt(6)) and r["gmr"] == pytest.approx(np.exp(v.mean()))

    def picks(rule, scope, task="a"):
        cells = res[rule, scope]["per_task"][task]["cells"]
        return [(c["seed"], c["selected"].split("::")[1]) for c in cells], [c["error"] for c in cells]

    for rule in ("err", "rank"):
        check(res[rule, "global"], glob)
        check(res[rule, "pertask"], per)
        # Selecting on every seed would pick p three times, on the held-out seed alone p, q, p.
        assert picks(rule, "pertask") == ([(20, "q"), (21, "p"), (22, "q")], pytest.approx([.05, .20, .05]))
        assert picks(rule, "global") == ([(20, "q"), (21, "q"), (22, "q")], pytest.approx([.05, .05, .05]))
        assert picks(rule, "pertask", "b")[0] == [(20, "q"), (21, "q"), (22, "q")]
    assert res["err", "pertask"]["difference_vs_first"]["mean_log_ratio"] == pytest.approx(np.mean(per) - np.mean(glob))
    assert rep["first"] == {"rule": "err", "scope": "global"} and "difference_vs_first" not in res["err", "global"]


def test_loso_compare_mode_and_prereg_refusal(tmp_path):
    stage, out = write_stage(tmp_path / "stage", loso_rows()), tmp_path / "cmp.json"
    loso(loso_args([stage], out, incumbent="tuned::ref", challengers=["tuned::q", "tuned::p"]))
    rep = json.loads(out.read_text())
    eps = .5 / 40
    v = np.array([np.log((.05 + eps) / (.2 + eps))] * 3 + [np.log(x / .5) for x in (.4, .5, .6)])
    q = rep["results"][0]
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(6)
    half = stats.t.ppf(.9, 5) * se
    assert rep["mode"] == "compare" and q["challenger"] == "tuned::q" and q["mean_log_ratio"] == pytest.approx(m)
    assert q["interval_80"] == pytest.approx([m - half, m + half]) and len(rep["results"]) == 2
    with pytest.raises(SystemExit, match=r"40 \(evaluation\)"):
        loso(loso_args([stage], out, seeds=[20, 21, 40], prereg=PREREG, reference="tuned::ref"))
    late = write_stage(tmp_path / "late", [full_row("a", "p", 35, .1)])
    with pytest.raises(SystemExit, match=r"35 \(reserved\)"):
        loso(loso_args([stage, late], out, prereg=PREREG, reference="tuned::ref"))
